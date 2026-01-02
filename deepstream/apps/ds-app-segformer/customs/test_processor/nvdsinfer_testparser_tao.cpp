#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <iostream>

// DeepStream includes
#include "nvdsinfer_custom_impl.h"  // Define NvDsInferLayerInfo
#include "nvdsinfer_context.h"      // Define NvDsInferContextHandle
#include "cuda_runtime_api.h"

extern "C" bool
NvDsInferParseSegTest(NvDsInferContextHandle context,
                      NvDsInferLayerInfo *output_layers_info,
                      NvDsInferSegmentationOutput *segOutput,
                      uint32_t num_classes)
{
    if (!output_layers_info || !segOutput) {
        std::cerr << "Output layers or segOutput is null!" << std::endl;
        return false;
    }

    float* outData = (float*)output_layers_info[0].buffer;

    // Dimensiones CHW o HWC según tu configuración
    int channels = output_layers_info[0].inferDims.numDims > 0 ? output_layers_info[0].inferDims.d[0] : 1;
    int height   = output_layers_info[0].inferDims.numDims > 1 ? output_layers_info[0].inferDims.d[1] : 1;
    int width    = output_layers_info[0].inferDims.numDims > 2 ? output_layers_info[0].inferDims.d[2] : 1;

    float maxVal = outData[0];
    float minVal = outData[0];
    int maxIdx = 0;

    for (int i = 0; i < width*height*channels; i++) {
        if (outData[i] > maxVal) { maxVal = outData[i]; maxIdx = i; }
        if (outData[i] < minVal) minVal = outData[i];
    }

    printf("[TestParser] Segformer inference: max=%.3f min=%.3f maxIdx=%d\n",
           maxVal, minVal, maxIdx);

    return true;
}
