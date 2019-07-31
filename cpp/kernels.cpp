/*
 * Copyright 2019 Vrije Universiteit Amsterdam, The Netherlands
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include "reduction/reduction.h"
#include "timer.h"

void reduceCPUKernel(const int n, float *sum, float *input) {
  *sum = 0.0;
  for (int i = 0; i < n; i++) {
    *sum += input[i] + 1;
  }
}


void reduceCPU(int n, float *sum, float *input) {
    timer t("cpu");

    t.start();
    reduceCPUKernel(n, sum, input);
    t.stop();

    std::cerr << t << std::endl;
}


void reduceMCL(int n, float *sum, float *input) {
  timer t("mcl");

  t.start();
  reduce(n, sum, input);
  t.stop();
  
  std::cerr << t << std::endl;
}
