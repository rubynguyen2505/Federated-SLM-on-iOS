#import "tensorflow/lite/c/c_api.h"

__attribute__((constructor))
static void ForceLinkTFLiteC() {
    // This forces the TfLiteModelCreate symbol to be included
    TfLiteModelCreate(NULL, 0);
}
