#include <omp.h>
#include <cstdio>
#include <string>
#include "main.h"
#include "version.h"
#include "OcrLite.h"

void printHelp(FILE *out, char *argv0) {
    fprintf(out, " ------- Usage -------\n");
    fprintf(out, "%s %s", argv0, usageMsg);
    fprintf(out, " ------- Required Parameters -------\n");
    fprintf(out, "%s", requiredMsg);
    fprintf(out, " ------- Optional Parameters -------\n");
    fprintf(out, "%s", optionalMsg);
    fprintf(out, " ------- Other Parameters -------\n");
    fprintf(out, "%s", otherMsg);
    fprintf(out, " ------- Examples -------\n");
    fprintf(out, example1Msg, argv0);
    fprintf(out, example2Msg, argv0);
}

int main(int argc, char **argv) {
    if (argc <= 1) {
        printHelp(stderr, argv[0]);
        return -1;
    }
    std::string modelsDir = "../models";
    std::string modelDetPath, modelClsPath, modelRecPath, keysPath;
    std::string argImgPath = "../../test_imgs/1.jpg";
    std::string imgPath, imgName;
    int numThread = 4;
    int padding = 0;
    int maxSideLen = 1024;
    float boxScoreThresh = 0.5f;
    float boxThresh = 0.3f;
    float unClipRatio = 2.0f;
    bool doAngle = true;
    int flagDoAngle = 1;
    bool mostAngle = false;
    int flagMostAngle = 0;

    int opt;
    int optionIndex = 0;
    while ((opt = getopt_long(argc, argv, "d:1:2:3:4:i:t:p:s:b:o:u:a:A:v:h", long_options, &optionIndex)) != -1) {
        //printf("option(-%c)=%s\n", opt, optarg);
        switch (opt) {
            case 'd':
                modelsDir = optarg;
                printf("modelsPath=%s\n", modelsDir.c_str());
                break;
            case '1':
                modelDetPath = modelsDir + "/" + optarg;
                printf("model det path=%s\n", modelDetPath.c_str());
                break;
            case '2':
                modelClsPath = modelsDir + "/" + optarg;
                printf("model cls path=%s\n", modelClsPath.c_str());
                break;
            case '3':
                modelRecPath = modelsDir + "/" + optarg;
                printf("model rec path=%s\n", modelRecPath.c_str());
                break;
            case '4':
                keysPath = modelsDir + "/" + optarg;
                printf("keys path=%s\n", keysPath.c_str());
                break;
            case 'i':
                argImgPath.assign(optarg);
                imgPath.assign(argImgPath.substr(0, argImgPath.find_last_of('/') + 1));
                imgName.assign(argImgPath.substr(argImgPath.find_last_of('/') + 1));
                printf("imgPath=%s, imgName=%s\n", imgPath.c_str(), imgName.c_str());
                break;
            case 't':
                numThread = (int) strtol(optarg, NULL, 10);
                //printf("numThread=%d\n", numThread);
                break;
            case 'p':
                padding = (int) strtol(optarg, NULL, 10);
                //printf("padding=%d\n", padding);
                break;
            case 's':
                maxSideLen = (int) strtol(optarg, NULL, 10);
                //printf("maxSideLen=%d\n", maxSideLen);
                break;
            case 'b':
                boxScoreThresh = strtof(optarg, NULL);
                //printf("boxScoreThresh=%f\n", boxScoreThresh);
                break;
            case 'o':
                boxThresh = strtof(optarg, NULL);
                //printf("boxThresh=%f\n", boxThresh);
                break;
            case 'u':
                unClipRatio = strtof(optarg, NULL);
                //printf("unClipRatio=%f\n", unClipRatio);
                break;
            case 'a':
                flagDoAngle = (int) strtol(optarg, NULL, 10);
                if (flagDoAngle == 0) {
                    doAngle = false;
                } else {
                    doAngle = true;
                }
                //printf("doAngle=%d\n", doAngle);
                break;
            case 'A':
                flagMostAngle = (int) strtol(optarg, NULL, 10);
                if (flagMostAngle == 0) {
                    mostAngle = false;
                } else {
                    mostAngle = true;
                }
                //printf("mostAngle=%d\n", mostAngle);
                break;
            case 'v':
                printf("%s\n", VERSION);
                return 0;
            case 'h':
                printHelp(stdout, argv[0]);
                return 0;
            default:
                printf("other option %c :%s\n", opt, optarg);
        }
    }
    omp_set_num_threads(numThread);
    OcrLite ocrLite;
    ocrLite.setNumThread(numThread);
    ocrLite.initLogger(
            true,//isOutputConsole
            false,//isOutputPartImg
            true);//isOutputResultImg

    ocrLite.enableResultTxt(imgPath.c_str(), imgName.c_str());
    ocrLite.Logger("=====Input Params=====\n");
    ocrLite.Logger(
            "numThread(%d),padding(%d),maxSideLen(%d),boxScoreThresh(%f),boxThresh(%f),unClipRatio(%f),doAngle(%d),mostAngle(%d)\n",
            numThread, padding, maxSideLen, boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle);

    ocrLite.initModels(modelDetPath, modelClsPath, modelRecPath, keysPath);

    OcrResult result = ocrLite.detect(imgPath.c_str(), imgName.c_str(), padding, maxSideLen,
                                      boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle);
    ocrLite.Logger("%s\n", result.strRes.c_str());
    return 0;
}
