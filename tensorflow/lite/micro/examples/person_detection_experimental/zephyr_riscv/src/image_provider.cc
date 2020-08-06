/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/micro/examples/person_detection_experimental/image_provider.h"
#include "tensorflow/lite/micro/examples/person_detection_experimental/model_settings.h"

// Temporary image provider (feed the image same from test)
#include "tensorflow/lite/micro/examples/person_detection_experimental/person_image_data.h"
// #include "tensorflow/lite/micro/examples/person_detection_experimental/no_person_image_data.h"

#include <cstring>

// Get an image from the camera module
TfLiteStatus GetImage(tflite::ErrorReporter* error_reporter, int image_width,
                      int image_height, int channels, int8_t* image_data) {
  // for (int i = 0; i < image_width * image_height * channels; ++i) {
  //   image_data[i] = 0;
  // }
  static bool feed_person = true;

  if (feed_person) {
    std::memcpy(image_data, g_person_data, g_person_data_size);
  }
  else {
    // std::memcpy(image_data, no_g_person_data, g_no_person_data_size);
    std::memset(image_data, 0, g_person_data_size);
  }

  feed_person = (!feed_person);

  return kTfLiteOk;
}
