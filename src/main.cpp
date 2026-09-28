#include <Arduino.h>
#include <WiFi.h>
#include "esp_camera.h"
#include "img_converters.h"
#include "esp_http_server.h"
#include <DNSServer.h>
#include "esp_system.h"
#include "config.h"

// ==============================================================================
// 1. TFLITE MICRO SETUP
// ==============================================================================
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"

constexpr int kTensorArenaSize = 4 * 1024 * 1024;
uint8_t *tensor_arena = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;

// ขนาดภาพที่โมเดลต้องการ (อ่านจาก tensor จริงตอน setupAI)
int model_w = 96;
int model_h = 96;

// บัฟเฟอร์สำหรับภาพ JPEG ที่ถอดรหัสแล้ว (RGB888 เต็มเฟรม) — จองใน PSRAM ครั้งเดียว
uint8_t *rgb_buf = nullptr;
size_t rgb_buf_size = 0;

// fmt2rgb888() ของ esp32-camera ให้ข้อมูลเรียงแบบ B,G,R
// ถ้าทดสอบแล้วสีดูสลับ (ผลทำนายเพี้ยนตามสี) ให้เปลี่ยนเป็น 0
#define DECODED_IS_BGR 1

const char *class_names[] = {"overripe", "ripe", "unripe"};

struct InferenceResult
{
  char best_class[16];
  float probabilities[3]; // [0] = overripe, [1] = ripe, [2] = unripe
  uint32_t infer_ms;
};
InferenceResult latest_result = {"Waiting...", {0.0f, 0.0f, 0.0f}, 0};
portMUX_TYPE result_mux = portMUX_INITIALIZER_UNLOCKED;

// ==============================================================================
// 2. WEB SERVER & HTML
// ==============================================================================
static const char INDEX_HTML[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>)rawliteral" PROJECT_TITLE R"rawliteral(</title>
  <style>
    body { font-family: sans-serif; background: #121212; color: #fff; text-align: center; margin: 0; padding: 20px; }
    .card { background: #1e1e1e; max-width: 400px; margin: auto; border-radius: 12px; padding: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.5); }
    img { width: 100%; height: auto; border-radius: 8px; border: 2px solid #333; background: #000; }
    .pred { font-size: 1.5rem; font-weight: bold; color: #4CAF50; margin: 15px 0 5px 0; text-transform: uppercase; }
    .bar-wrap { background: #333; border-radius: 6px; overflow: hidden; height: 16px; margin: 4px 0 10px 0; }
    .bar { height: 100%; width: 0%; transition: width 0.2s; }
    .label { display: flex; justify-content: space-between; font-size: 0.85rem; color: #ccc; }
    .stats { font-size: 0.75rem; color: #777; margin-top: 15px; border-top: 1px solid #333; padding-top: 10px; }
  </style>
</head>
<body>
  <div class="card">
    <h3>)rawliteral" PROJECT_TITLE " - " GROUP_NAME R"rawliteral(</h3>
    <img id="stream" src="http://192.168.4.1:81/stream" alt="Live Feed">
    <div id="pred" class="pred">Waiting...</div>

    <div class="label"><span>Unripe</span><span id="p-unripe">0%</span></div>
    <div class="bar-wrap"><div id="b-unripe" class="bar" style="background: #FFC107;"></div></div>

    <div class="label"><span>Ripe</span><span id="p-ripe">0%</span></div>
    <div class="bar-wrap"><div id="b-ripe" class="bar" style="background: #4CAF50;"></div></div>

    <div class="label"><span>Overripe</span><span id="p-overripe">0%</span></div>
    <div class="bar-wrap"><div id="b-overripe" class="bar" style="background: #F44336;"></div></div>

    <div class="stats" id="stats">AI Status: Active</div>
  </div>

  <script>
    function setBar(name, v) {
      document.getElementById('p-' + name).innerText = v.toFixed(1) + '%';
      document.getElementById('b-' + name).style.width = Math.max(0, Math.min(100, v)) + '%';
    }
    setInterval(() => {
      fetch('/result?_=' + Date.now(), { cache: 'no-store' })
        .then(r => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
        .then(d => {
          document.getElementById('pred').innerText = d.class;
          setBar('unripe', d.unripe);
          setBar('ripe', d.ripe);
          setBar('overripe', d.overripe);
          document.getElementById('stats').innerText = 'AI Status: Active | Inference: ' + d.ms + ' ms';
        })
        .catch(e => {
          document.getElementById('stats').innerText = 'AI Status: Disconnected';
        });
    }, 1000);
  </script>
</body>
</html>
)rawliteral";

static esp_err_t index_handler(httpd_req_t *req)
{
  httpd_resp_set_type(req, "text/html");
  return httpd_resp_send(req, INDEX_HTML, strlen(INDEX_HTML));
}

esp_err_t captive_portal_handler(httpd_req_t *req, httpd_err_code_t err)
{
  httpd_resp_set_status(req, "302 Found");
  httpd_resp_set_hdr(req, "Location", "http://192.168.4.1/");
  httpd_resp_send(req, NULL, 0);
  return ESP_OK;
}

static esp_err_t result_handler(httpd_req_t *req)
{
  InferenceResult r;
  portENTER_CRITICAL(&result_mux);
  r = latest_result; // คัดลอกออกมาก่อน กันข้อมูลถูกเขียนทับกลางทาง
  portEXIT_CRITICAL(&result_mux);

  char json[256];
  snprintf(json, sizeof(json),
           "{\"class\":\"%s\",\"overripe\":%.1f,\"ripe\":%.1f,\"unripe\":%.1f,\"ms\":%u}",
           r.best_class,
           r.probabilities[0] * 100.0f,
           r.probabilities[1] * 100.0f,
           r.probabilities[2] * 100.0f,
           (unsigned)r.infer_ms);

  httpd_resp_set_type(req, "application/json");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  httpd_resp_set_hdr(req, "Cache-Control", "no-store");
  return httpd_resp_send(req, json, strlen(json));
}

static esp_err_t stream_handler(httpd_req_t *req)
{
  camera_fb_t *fb = NULL;
  esp_err_t res = ESP_OK;
  char part_buf[128];

  res = httpd_resp_set_type(req, "multipart/x-mixed-replace;boundary=123456789000000000000987654321");
  if (res != ESP_OK)
    return res;

  while (true)
  {
    fb = esp_camera_fb_get();
    if (!fb)
    {
      res = ESP_FAIL;
      break;
    }

    size_t hlen = snprintf(part_buf, sizeof(part_buf),
                           "\r\n--123456789000000000000987654321\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n",
                           (unsigned)fb->len);
    res = httpd_resp_send_chunk(req, part_buf, hlen);
    if (res == ESP_OK)
      res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);

    esp_camera_fb_return(fb);
    fb = NULL;

    if (res != ESP_OK)
      break;
    delay(1); // ให้ task อื่นได้ใช้ CPU บ้าง
  }
  return res;
}

httpd_handle_t ui_httpd = NULL;     // Port 80: หน้าเว็บ + JSON
httpd_handle_t stream_httpd = NULL; // Port 81: วิดีโอสตรีม

void startWebServer()
{
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  config.max_uri_handlers = 4;

  httpd_uri_t index_uri = {.uri = "/", .method = HTTP_GET, .handler = index_handler, .user_ctx = NULL};
  httpd_uri_t result_uri = {.uri = "/result", .method = HTTP_GET, .handler = result_handler, .user_ctx = NULL};

  if (httpd_start(&ui_httpd, &config) == ESP_OK)
  {
    httpd_register_uri_handler(ui_httpd, &index_uri);
    httpd_register_uri_handler(ui_httpd, &result_uri);
    httpd_register_err_handler(ui_httpd, HTTPD_404_NOT_FOUND, captive_portal_handler);
  }

  httpd_config_t stream_config = HTTPD_DEFAULT_CONFIG();
  stream_config.server_port = 81;
  stream_config.ctrl_port = 32769; // ต้องไม่ซ้ำกับ server แรก (ค่า default คือ 32768)
  httpd_uri_t stream_uri = {.uri = "/stream", .method = HTTP_GET, .handler = stream_handler, .user_ctx = NULL};

  if (httpd_start(&stream_httpd, &stream_config) == ESP_OK)
  {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
  }
}

// ==============================================================================
// 3. AI INFERENCE LOGIC
// ==============================================================================
void setupAI()
{
  Serial.println("[AI] Allocating Tensor Arena in PSRAM...");
  tensor_arena = (uint8_t *)ps_malloc(kTensorArenaSize);
  if (!tensor_arena)
  {
    Serial.println("[AI ERROR] PSRAM Allocation Failed for Tensor Arena!");
    return;
  }

  static tflite::MicroErrorReporter micro_error_reporter;
  const tflite::Model *model = tflite::GetModel(mangosteen_quant_int8_tflite);
  if (!model)
  {
    Serial.println("[AI ERROR] Model pointer is null!");
    return;
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk)
  {
    Serial.println("[AI ERROR] AllocateTensors failed!");
    return;
  }

  TfLiteTensor *in = interpreter->input(0);
  TfLiteTensor *out = interpreter->output(0);

  // ตรวจรูปร่าง tensor: input ต้องเป็น [1, H, W, 3] ชนิด int8, output ต้องมี 3 คลาส
  if (in->type != kTfLiteInt8 || in->dims->size != 4 || in->dims->data[3] != 3)
  {
    Serial.println("[AI ERROR] Unexpected input tensor shape/type!");
    return;
  }
  if (out->type != kTfLiteInt8 || out->dims->data[out->dims->size - 1] != 3)
  {
    Serial.println("[AI ERROR] Unexpected output tensor shape/type!");
    return;
  }

  model_h = in->dims->data[1];
  model_w = in->dims->data[2];
  input = in;
  output = out;
  Serial.printf("[AI OK] TFLite ready. Input %dx%dx3\n", model_w, model_h);
}

// เดิม: โค้ดอ่าน fb->buf เป็นภาพ RGB565 ขนาด 96x96 แต่กล้องตั้งเป็น JPEG 640x480
//       โมเดลจึงได้ไบต์ของไฟล์ JPEG ไม่ใช่พิกเซลจริง -> ผลทำนายเพี้ยน
// แก้:  ถอดรหัส JPEG -> RGB888, ครอปตรงกลางให้เป็นสี่เหลี่ยมจัตุรัส, ย่อเป็นขนาดที่โมเดลต้องการ
void runAIInference()
{
  if (input == nullptr || output == nullptr)
    return;

  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb)
    return;

  uint32_t t0 = millis();
  const int src_w = fb->width;
  const int src_h = fb->height;

  // จองบัฟเฟอร์ RGB888 ครั้งแรก (640*480*3 ≈ 900 KB ใน PSRAM)
  size_t need = (size_t)src_w * src_h * 3;
  if (rgb_buf == nullptr || rgb_buf_size < need)
  {
    if (rgb_buf)
      free(rgb_buf);
    rgb_buf = (uint8_t *)ps_malloc(need);
    rgb_buf_size = rgb_buf ? need : 0;
    if (!rgb_buf)
    {
      Serial.println("[AI ERROR] PSRAM Allocation Failed for RGB buffer!");
      esp_camera_fb_return(fb);
      return;
    }
  }

  bool ok = fmt2rgb888(fb->buf, fb->len, fb->format, rgb_buf);
  esp_camera_fb_return(fb); // คืนเฟรมให้กล้องเร็วที่สุด สตรีมจะได้ไม่สะดุด
  if (!ok)
  {
    Serial.println("[AI ERROR] JPEG decode failed");
    return;
  }

  // ครอปตรงกลางเป็นสี่เหลี่ยมจัตุรัส แล้วย่อแบบ nearest-neighbor
  const int crop = min(src_w, src_h);
  const int off_x = (src_w - crop) / 2;
  const int off_y = (src_h - crop) / 2;

  int8_t *input_buffer = input->data.int8;
  const float input_scale = input->params.scale;
  const int input_zero_point = input->params.zero_point;

  int idx = 0;
  for (int y = 0; y < model_h; y++)
  {
    int sy = off_y + (y * crop) / model_h;
    for (int x = 0; x < model_w; x++)
    {
      int sx = off_x + (x * crop) / model_w;
      const uint8_t *p = &rgb_buf[(sy * src_w + sx) * 3];
#if DECODED_IS_BGR
      uint8_t r = p[2], g = p[1], b = p[0];
#else
      uint8_t r = p[0], g = p[1], b = p[2];
#endif
      // normalize เป็น 0.0 ถึง 1.0 (pixel / 255)
      // ตรงกับค่า quantization ของ input tensor ในโมเดล (scale = 1/255, zero_point = -128)
      // หมายเหตุ: เวอร์ชันเดิม normalize เป็น -1..1 ซึ่งไม่ตรงกับโมเดล ทำให้ค่าเกินช่วงและถูกตัดทิ้ง
      float nr = (float)r / 255.0f;
      float ng = (float)g / 255.0f;
      float nb = (float)b / 255.0f;

      input_buffer[idx++] = (int8_t)constrain(lroundf(nr / input_scale) + input_zero_point, -128, 127);
      input_buffer[idx++] = (int8_t)constrain(lroundf(ng / input_scale) + input_zero_point, -128, 127);
      input_buffer[idx++] = (int8_t)constrain(lroundf(nb / input_scale) + input_zero_point, -128, 127);
    }
  }

  if (interpreter->Invoke() != kTfLiteOk)
  {
    Serial.println("[AI ERROR] Invoke failed");
    return;
  }

  const int8_t *out_data = output->data.int8;
  const float scale = output->params.scale;
  const int zero_point = output->params.zero_point;

  int best_idx = 0;
  float probs[3];
  for (int i = 0; i < 3; i++)
  {
    probs[i] = (out_data[i] - zero_point) * scale;
    if (probs[i] > probs[best_idx])
      best_idx = i;
  }
  uint32_t dt = millis() - t0;

  portENTER_CRITICAL(&result_mux);
  strncpy(latest_result.best_class, class_names[best_idx], sizeof(latest_result.best_class) - 1);
  latest_result.best_class[sizeof(latest_result.best_class) - 1] = '\0';
  for (int i = 0; i < 3; i++)
    latest_result.probabilities[i] = probs[i];
  latest_result.infer_ms = dt;
  portEXIT_CRITICAL(&result_mux);

  Serial.printf("[AI] %s | overripe %.2f ripe %.2f unripe %.2f | %u ms\n",
                class_names[best_idx], probs[0], probs[1], probs[2], (unsigned)dt);
}

// ==============================================================================
// 4. SETUP & MAIN LOOP
// ==============================================================================
const byte DNS_PORT = 53;
DNSServer dnsServer;
bool wifi_started = false;

void setup()
{
  Serial.begin(115200);
  delay(1000);

  pinMode(PWR_ON_PIN, OUTPUT);
  digitalWrite(PWR_ON_PIN, HIGH);
  delay(150);

  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = CAM_Y2_PIN;
  config.pin_d1 = CAM_Y3_PIN;
  config.pin_d2 = CAM_Y4_PIN;
  config.pin_d3 = CAM_Y5_PIN;
  config.pin_d4 = CAM_Y6_PIN;
  config.pin_d5 = CAM_Y7_PIN;
  config.pin_d6 = CAM_Y8_PIN;
  config.pin_d7 = CAM_Y9_PIN;
  config.pin_xclk = CAM_XCLK_PIN;
  config.pin_pclk = CAM_PCLK_PIN;
  config.pin_vsync = CAM_VSYNC_PIN;
  config.pin_href = CAM_HREF_PIN;
  config.pin_sccb_sda = CAM_SIOD_PIN;
  config.pin_sccb_scl = CAM_SIOC_PIN;
  config.pin_pwdn = CAM_PWDN_PIN;
  config.pin_reset = CAM_RESET_PIN;

  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_VGA;
  config.jpeg_quality = 12;
  config.fb_count = 2;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_LATEST; // ใช้เฟรมล่าสุดเสมอ ไม่ใช่เฟรมค้างในคิว

  if (esp_camera_init(&config) != ESP_OK)
  {
    Serial.println("[CAM ERROR] Camera init failed");
    while (true)
      delay(1000);
  }
  Serial.println("[CAM OK] Camera initialized");

  setupAI();

  WiFi.mode(WIFI_AP);
  WiFi.softAP(WIFI_AP_SSID, WIFI_AP_PASSWORD);
  Serial.printf("[WIFI] AP \"%s\" started, open http://%s/\n",
                WIFI_AP_SSID, WiFi.softAPIP().toString().c_str());

  dnsServer.start(DNS_PORT, "*", WiFi.softAPIP());
  wifi_started = true;

  startWebServer();
}

void loop()
{
  static uint32_t last_infer = 0;

  if (wifi_started)
    dnsServer.processNextRequest();

  // รันโมเดลทุก 2 วินาที โดยไม่บล็อก DNS server
  if (millis() - last_infer >= 2000)
  {
    last_infer = millis();
    runAIInference();
  }
  delay(10);
}
