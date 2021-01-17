import os
import onnx

def WalkDir(dirlist,filelist,dirname):
    try:
        ls=os.listdir(dirname)
    except:
         print(" Access Deny.")
    else:
        for fn in ls:
            temp=os.path.join(dirname,fn)
            if (os.path.isdir(temp)):
               dirlist.append(temp)
              # WalkDir(dirlist,filelist,temp)
            else:
                if temp.endswith(".onnx"):
                    filelist.append(temp)
    return dirlist,filelist

def ConvertModel(srcfile, dstfile):
    onnx_model = onnx.load(srcfile)
    onnx_model.graph.input[0].type.tensor_type.shape.dim[2].dim_param = '?'  # w涓哄姩鎬佽緭鍏?
    onnx_model.graph.input[0].type.tensor_type.shape.dim[3].dim_param = '?'
    onnx.save(onnx_model, dstfile)


dirlist=[]
filelist=[]
WalkDir(dirlist,filelist,os.getcwd())

for file in filelist:
    dstfile=file+".new"
    ConvertModel(file,dstfile)
