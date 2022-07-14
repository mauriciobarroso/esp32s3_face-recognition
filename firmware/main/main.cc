/**
  ******************************************************************************
  * @file           : main.cc
  * @author         : Mauricio Barroso Benavides
  * @date           : Jul 10, 2022
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * MIT License
  *
  * Copyright (c) 2022 Mauricio Barroso Benavides
  *
  * Permission is hereby granted, free of charge, to any person obtaining a copy
  * of this software and associated documentation files (the "Software"), to
  * deal in the Software without restriction, including without limitation the
  * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
  * sell copies of the Software, and to permit persons to whom the Software is
  * furnished to do so, subject to the following conditions:
  *
  * The above copyright notice and this permission notice shall be included in
  * all copies or substantial portions of the Software.
  *
  * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
  * IN THE SOFTWARE.
  *
  ******************************************************************************
  */

/* Includes ------------------------------------------------------------------*/
#include "esp_log.h"
#include "esp_system.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "wifi.h"
#include "led_strip.h"
#include "camera.h"
#include "mqtt_client.h"
#include "image_utils.c"

#include "models/face_detection_model_settings.h"
#include "models/face_detection_model_data.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

/* Private typedef -----------------------------------------------------------*/

/* Private macro -------------------------------------------------------------*/
#define BLINK_GPIO 48
#define BLINK_LED_RMT_CHANNEL 0
#define MQTT_BROKER_URL	"mqtts://a4bdl1178z00a-ats.iot.us-east-1.amazonaws.com:8883"
#define SCRATH_BUFFER_SIZE (39 * 1024)
#define TENSOR_ARENA_SIZE (81 * 1024 + SCRATH_BUFFER_SIZE)

/* Private variables ---------------------------------------------------------*/
static const char * TAG = "app"; /* Tag for debugging */
static wifi_t wifi;
static esp_mqtt_client_handle_t mqtt_client;

/* Server certificate*/
extern const uint8_t server_cert_pem_start[] asm("_binary_ca_pem_start");
extern const uint8_t server_cert_pem_end[] asm("_binary_ca_pem_end");

/* Thing certificate */
extern const uint8_t client_cert_pem_start[] asm("_binary_certificate_pem_crt_start");
extern const uint8_t client_cert_pem_end[] asm("_binary_certificate_pem_crt_end");

/* Private key */
extern const uint8_t client_key_pem_start[] asm("_binary_private_pem_key_start");
extern const uint8_t client_key_pem_end[] asm("_binary_private_pem_key_end");

static uint8_t * tensor_arena;
static led_strip_t * led;
tflite::ErrorReporter * error_reporter = nullptr;
const tflite::Model * face_detection_model = nullptr;
tflite::MicroInterpreter * face_detection_interpreter = nullptr;
TfLiteTensor * face_detection_input = nullptr;

/* Private function prototypes -----------------------------------------------*/
static void inference_task(void * arg);
static void tflm_init(void);
static esp_err_t nvs_init(void);
static void led_init(void);
static esp_err_t mqtt_init(void);

/* Event handlers */
static void wifi_event_handler(void * arg, esp_event_base_t event_base, int32_t event_id, void * event_data);
static void ip_event_handler(void * arg, esp_event_base_t event_base, int32_t event_id, void * event_data);
static void prov_event_handler(void * arg, esp_event_base_t event_base, int32_t event_id, void * event_data);
static void mqtt_event_handler(void * arg, esp_event_base_t event_base, int32_t event_id, void * event_data);

/* Utils */
static void responder_to_detection(tflite::ErrorReporter * error_reporter, float face_score);
static TfLiteStatus get_image(tflite::ErrorReporter* error_reporter, int image_width, int image_height, int channels, int8_t * image_data);
/* Private user code ---------------------------------------------------------*/
extern "C" void app_main() {
    /* Initialize componentes */
	ESP_LOGI(TAG, "Initializing components..");
	led_init();
    camera_init();
	nvs_init();

	wifi.wifi_event_handler= wifi_event_handler;
	wifi.ip_event_handler = ip_event_handler;
	wifi.prov_event_handler = prov_event_handler;
	wifi_init(&wifi);
	mqtt_init();

	/* Initialize TFLM */
	tflm_init();

	/* Create RTOS tasks */
	xTaskCreate(inference_task,
			"Inference Task",
			configMINIMAL_STACK_SIZE * 8,
			NULL,
			tskIDLE_PRIORITY + 8,
			NULL);
}

static void inference_task(void * arg) {
	for(;;) {
		/* Get image */
		if(kTfLiteOk != get_image(error_reporter,
				NUM_COLS,
				NUM_ROWS,
				NUM_CHANNELS,
				face_detection_input->data.int8)) {
			TF_LITE_REPORT_ERROR(error_reporter, "Image capture failed.");
		}

		/* Run the model on this input and make sure it succeeds */
		if(kTfLiteOk != face_detection_interpreter->Invoke()) {
			TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
		}

		TfLiteTensor * output = face_detection_interpreter->output(0);

		/* Process the inference results */
		int8_t face_score_int = output->data.uint8[FACE_INDEX];

		float face_score_float = (face_score_int - output->params.zero_point) * output->params.scale;

		/* Respond to detection */
		responder_to_detection(error_reporter, face_score_float);

		vTaskDelay(20); /* to avoid watchdog trigger */
	}
}

static esp_err_t nvs_init(void) {
	esp_err_t ret;

	ret = nvs_flash_init();

    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        /* NVS partition was truncated and needs to be erased.Retry
         * nvs_flash_init */
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }

	return ret;
}

static void led_init(void) {
    ESP_LOGI(TAG, "Example configured to blink addressable LED!");
    /* LED strip initialization with the GPIO and pixels number*/
    led = led_strip_init(0, BLINK_GPIO, 1);
    /* Set all LED off to clear all pixels */
    led->clear(led, 50);
}

static esp_err_t mqtt_init(void) {
	esp_err_t ret;

	ESP_LOGI(TAG, "Initializing MQTT...");

	/* Fill MQTT client configuration and initialize */
	esp_mqtt_client_config_t mqtt_config = {
			.uri = MQTT_BROKER_URL,
			.cert_pem = (const char *)server_cert_pem_start,
			.client_cert_pem = (const char *)client_cert_pem_start,
			.client_key_pem = (const char *)client_key_pem_start,
	};

	mqtt_client = esp_mqtt_client_init(&mqtt_config);

	/* Register MQTT event handler */
	if(mqtt_client != NULL) {
		ret = esp_mqtt_client_register_event(mqtt_client,
				MQTT_EVENT_ANY,
				mqtt_event_handler,
				NULL);

		if(ret != ESP_OK) {
			return ret;
		}
	}
	else {
		return ESP_FAIL;
	}

	return ret;
}

static void tflm_init(void) {
	static tflite::MicroErrorReporter micro_error_reporter;
	error_reporter = &micro_error_reporter;

	face_detection_model = tflite::GetModel(g_face_detection_model_data);
	if(face_detection_model->version() != TFLITE_SCHEMA_VERSION) {
		TF_LITE_REPORT_ERROR(error_reporter,
				"Model provided is schema version %d not equal "
				"to supported version %d.",
				face_detection_model->version(),
				TFLITE_SCHEMA_VERSION);
		return;
	}

	if(tensor_arena == NULL) {
		tensor_arena = (uint8_t *)heap_caps_malloc(TENSOR_ARENA_SIZE,
				MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
	}

	if (tensor_arena == NULL) {
		printf("Couldn't allocate memory of %d bytes\n", TENSOR_ARENA_SIZE);
		return;
	}

	/* Add operations needed */
	static tflite::MicroMutableOpResolver<5> resolver;
	resolver.AddAveragePool2D();
	resolver.AddConv2D();
	resolver.AddDepthwiseConv2D();
	resolver.AddReshape();
	resolver.AddSoftmax();

	/* Build an interpreter to run the model with */
	static tflite::MicroInterpreter face_detection_static_interpreter(
			face_detection_model,
			resolver,
			tensor_arena,
			TENSOR_ARENA_SIZE,
			error_reporter);

	face_detection_interpreter = &face_detection_static_interpreter;

	/* Allocate memory from the tensor_arena for the model's tensors */
	TfLiteStatus allocate_status = face_detection_interpreter->AllocateTensors();

	if(allocate_status != kTfLiteOk) {
		TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
		return;
	}

	/* Get information about the memory area to use for the model's input */
	face_detection_input = face_detection_interpreter->input(0);
}

/* Event handlers */
static void wifi_event_handler(void * arg, esp_event_base_t event_base, int32_t event_id, void * event_data) {
	switch(event_id) {
		case WIFI_EVENT_STA_CONNECTED: {
			ESP_LOGI(TAG, "WIFI_EVENT_STA_CONNECTED");

			break;
		}

		case WIFI_EVENT_STA_DISCONNECTED: {
			ESP_LOGI(TAG, "WIFI_EVENT_STA_DISCONNECTED");

	        break;
		}

		case WIFI_EVENT_AP_STACONNECTED: {
			ESP_LOGI(TAG, "WIFI_EVENT_AP_STACONNECTED");

			break;
		}

		default:
			ESP_LOGI(TAG, "Other event");
			break;
	}
}

static void ip_event_handler(void * arg, esp_event_base_t event_base, int32_t event_id, void * event_data) {
	switch(event_id) {
		case IP_EVENT_STA_GOT_IP: {
			ESP_LOGI(TAG, "IP_EVENT_STA_GOT_IP");

			/* Start MQTT client */
			esp_mqtt_client_start(mqtt_client);

			break;
		}
		default: {
			ESP_LOGI(TAG, "Other event");

			break;
		}
	}
}

static void prov_event_handler(void * arg, esp_event_base_t event_base, int32_t event_id, void * event_data) {
	switch(event_id) {
		case WIFI_PROV_START: {
			ESP_LOGI(TAG, "WIFI_PROV_START");

			break;
		}

		case WIFI_PROV_CRED_RECV: {
			ESP_LOGI(TAG, "WIFI_PROV_CRED_RECV");

			wifi_sta_config_t * wifi_sta_cfg = (wifi_sta_config_t *)event_data;
			ESP_LOGI(TAG, "Credentials received, SSID: %s & Password: %s", (const char *) wifi_sta_cfg->ssid, (const char *) wifi_sta_cfg->password);

			break;
		}

		case WIFI_PROV_CRED_SUCCESS: {
			ESP_LOGI(TAG, "WIFI_PROV_CRED_SUCCESS");

			break;
		}

		case WIFI_PROV_END: {
			ESP_LOGI(TAG, "WIFI_PROV_END");

			/* De-initialize manager once provisioning is finished */
			wifi_prov_mgr_deinit();

			break;
		}

		case WIFI_PROV_CRED_FAIL: {
			ESP_LOGI(TAG, "WIFI_PROV_CRED_FAIL");

			/* Erase any stored Wi-Fi credentials  */
			ESP_LOGI(TAG, "Erasing Wi-Fi credentials");

			esp_err_t ret;

			nvs_handle_t nvs_handle;
			ret = nvs_open("nvs.net80211", NVS_READWRITE, &nvs_handle);

			if(ret == ESP_OK) {
				nvs_erase_all(nvs_handle);
			}

			/* Close NVS */
			ret = nvs_commit(nvs_handle);
			nvs_close(nvs_handle);

			if(ret == ESP_OK) {
				/* Restart device */
				esp_restart();
			}

			break;
		}

		case WIFI_PROV_DEINIT: {
			ESP_LOGI(TAG, "WIFI_PROV_DEINIT");

			/* Start Wi-Fi */
			ESP_ERROR_CHECK(esp_wifi_stop());
			ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
			ESP_ERROR_CHECK(esp_wifi_start());
			ESP_ERROR_CHECK(esp_wifi_connect());

			break;
		}
		default: {
			ESP_LOGI(TAG, "Other event");

			break;
		}
	}
}

static void mqtt_event_handler(void * arg, esp_event_base_t event_base,
		int32_t event_id, void * event_data) {

	switch(event_id) {
		case MQTT_EVENT_CONNECTED: {
			ESP_LOGI(TAG, "MQTT_EVENT_CONNECTED");

			esp_mqtt_client_publish(mqtt_client, "wfdata", "connected", 9, 0, 0);


			break;
		}

		case MQTT_EVENT_DISCONNECTED: {
			ESP_LOGI(TAG, "MQTT_EVENT_DISCONNECTED");

			break;
		}

		case MQTT_EVENT_DATA: {
			ESP_LOGI(TAG, "MQTT_EVENT_DATA");

			break;
		}

		default:
			ESP_LOGI(TAG, "Other MQTT event");
			break;
	}
}

static void responder_to_detection(tflite::ErrorReporter * error_reporter, float face_score) {
	int face_score_int = (face_score) * 100 + 0.5;
	static uint8_t face_counter = 0;

	/* To ensure a face is detected */
	if(face_score_int >= 95) {
		face_counter++;

		if(face_counter == 10) {
			ESP_LOGI(TAG, "Face detected!");

			led->set_pixel(led, 0, 45, 16, 16);
			led->refresh(led, 100);

			esp_mqtt_client_publish(mqtt_client, "wfdata", "face_detected", 13, 0, 0);
		}
	}
	else {
		face_counter = 0;
		/* Set all LED off to clear all pixels */
		led->clear(led, 50);
	}
}

static TfLiteStatus get_image(tflite::ErrorReporter * error_reporter, int image_width, int image_height, int channels, int8_t * image_data) {
	camera_fb_t * fb = esp_camera_fb_get();

	/* Check image captured state */
	if(!fb) {
		ESP_LOGE(TAG, "Camera capture failed");
		return kTfLiteError;
	}

	uint8_t * tmp_buffer = (uint8_t *) malloc(image_width * image_height * channels);

	image_resize_linear(tmp_buffer, fb->buf, image_width, image_height, channels, 240, 240);

	/* Quantize imaget to int8 */
	for (int i = 0; i < image_width * image_height * channels; i++) {
		image_data[i] = tmp_buffer ^ 0x80;
	}

	free(tmp_buffer);

	esp_camera_fb_return(fb);

	return kTfLiteOk;
}

/***************************** END OF FILE ************************************/
