#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <algorithm>
#include "nvdsinfer_custom_impl.h"
#include "nvdsinfer_context.h"
#include <glib.h>  // Para g_print

extern "C" bool
NvDsInferParseSegTest(NvDsInferContextHandle context,
                      NvDsInferLayerInfo *output_layers_info,
                      NvDsInferSegmentationOutput *segOutput,
                      uint32_t num_classes)
{
    if (!output_layers_info) {
        g_printerr("NvDsInferParseSegTest: output_layers_info is null!\n");
        return false;
    }

    float* outData = (float*)output_layers_info[0].buffer;

    int channels = output_layers_info[0].inferDims.numDims > 0 ? output_layers_info[0].inferDims.d[0] : 1;
    int height   = output_layers_info[0].inferDims.numDims > 1 ? output_layers_info[0].inferDims.d[1] : 1;
    int width    = output_layers_info[0].inferDims.numDims > 2 ? output_layers_info[0].inferDims.d[2] : 1;

    int wh = width * height;

    // Solo sanity check: argmax y min/max
    float maxVal = outData[0];
    float minVal = outData[0];
    int maxIdx = 0;

    for (int i = 0; i < wh; i++) {
        int max_class = 0;
        float score = outData[i];
        for (int c = 1; c < channels; c++) {
            float s = outData[c * wh + i]; // CHW
            if (s > score) {
                score = s;
                max_class = c;
            }
        }
        if (score > maxVal) { maxVal = score; maxIdx = i; }
        if (score < minVal) minVal = score;
    }

    g_print("[NvDsInferParseSegTest] Segformer tensor: %dx%d, classes=%d, max=%.3f min=%.3f maxIdx=%d\n",
            width, height, channels, maxVal, minVal, maxIdx);

    return true;
}
