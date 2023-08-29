import timm
import numpy as np
import sys
import onnxruntime
import torch
import io
import onnx
import onnx_graphsurgeon as gs
import time

def onnx_time_check(onnx_model):
    times = 100
    
    ort_sess = onnxruntime.InferenceSession(onnx_model, providers=['CUDAExecutionProvider'])    # gpu 측정
    # ort_sess = onnxruntime.InferenceSession(onnx_model, providers=['CPUExecutionProvider'])   # cpu 측정
    
    # print(torch.version.cuda)  # 현재 설치된 CUDA 버전 확인
    # print(torch.backends.cudnn.version())  # 현재 설치된 cuDNN 버전 확인
    # print(onnxruntime.get_device())  # cpu인지 gpu인지 확인

    input_name = ort_sess.get_inputs()[0].name
    input_shape = ort_sess.get_inputs()[0].shape

    #check input_shape[2] is str
    if input_shape[2] == "height" or input_shape[2] == "width":
        height, width = 640, 640
    else:
        height, width = input_shape[2], input_shape[3]

    # 배치에 따라 100번씩 인퍼런스 시켜서 평균을 측정
    time_list = []
    for batch_size in [1, 2, 4, 8, 16, 32, 64]:
        input = np.random.randn(batch_size, 3, height, width).astype(np.float32)

        start = time.time()
        for i in range(times):
            ort_sess.run(None, {input_name: input})
        end = time.time()
      
        # print(f"batch_size:{batch_size}, {round((end-start)/times, 5)} sec")
        time_list.append(str(round((end-start)/times, 5)))

    return time_list

def main():
    model_list = timm.list_models()
    error_list = []
    succese_list = []

    f1 = open('timm_seccese_model_list.txt', 'a')
    f1.write('model,batch1,batch2,batch4,batch8,batch16,batch32,batch64\n')
    f1.close()
  
    for one_model in model_list:
        try:
            model = timm.create_model(one_model, pretrained=False)
            model.eval()

            x = torch.randn(1, 3, 384, 128)

            # onnx export
            onnx_buff = io.BytesIO()
            torch.onnx.export(
                model, 
                x, 
                onnx_buff, 
                export_params=True, 
                # opset_version=10, 
                do_constant_folding=True, 
                input_names=['input'], 
                output_names=['output'], 
                dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}}
            )

            # onnx model 따로 저장하지 않고 bytes buffer로 넘기기
            model_proto = onnx.load_model_from_string(onnx_buff.getvalue())
            model_proto_bytes = onnx._serialize(model_proto)

            time_list = onnx_time_check(model_proto_bytes)

            f1 = open('timm_seccese_model_list.txt', 'a')
            f1.write(one_model + ',')
            f1.write(','.join(time_list) + '\n')    # 엑셀에서 보기 편하도록
            f1.close()

            print('[seccese] ', one_model)
            succese_list.append(one_model)

        except:
            print('[error] ', one_model)
            f2 = open('timm_error_model_list.txt', 'a')
            f2.write(one_model + '\n')
            f2.close()
            error_list.append(one_model)
            continue    


    print('==============')
    print('*** total model : ', len(timm.list_models()))
    print('*** error : ', len(error_list))
    print('*** error : ', len(error_list))


if __name__ == "__main__":
        main()
