import torch
import torch.nn as nn
import collections
import cupy
import re

'''
from the original repository of SoftSplat:
https://github.com/sniklaus/softmax-splatting
'''



kernel_Softsplat_updateOutput = '''
	extern "C" __global__ void kernel_Softsplat_updateOutput(
		const int n,
		const float* input,
		const float* flow,
		float* output
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
		const int intN = ( intIndex / SIZE_3(output) / SIZE_2(output) / SIZE_1(output) ) % SIZE_0(output);
		const int intC = ( intIndex / SIZE_3(output) / SIZE_2(output)                  ) % SIZE_1(output);
		const int intY = ( intIndex / SIZE_3(output)                                   ) % SIZE_2(output);
		const int intX = ( intIndex                                                    ) % SIZE_3(output);
		float fltOutputX = (float) (intX) + VALUE_4(flow, intN, 0, intY, intX);
		float fltOutputY = (float) (intY) + VALUE_4(flow, intN, 1, intY, intX);
		int intNorthwestX = (int) (floor(fltOutputX));
		int intNorthwestY = (int) (floor(fltOutputY));
		int intNortheastX = intNorthwestX + 1;
		int intNortheastY = intNorthwestY;
		int intSouthwestX = intNorthwestX;
		int intSouthwestY = intNorthwestY + 1;
		int intSoutheastX = intNorthwestX + 1;
		int intSoutheastY = intNorthwestY + 1;
		float fltNorthwest = ((float) (intSoutheastX) - fltOutputX   ) * ((float) (intSoutheastY) - fltOutputY   );
		float fltNortheast = (fltOutputX    - (float) (intSouthwestX)) * ((float) (intSouthwestY) - fltOutputY   );
		float fltSouthwest = ((float) (intNortheastX) - fltOutputX   ) * (fltOutputY    - (float) (intNortheastY));
		float fltSoutheast = (fltOutputX    - (float) (intNorthwestX)) * (fltOutputY    - (float) (intNorthwestY));
		if ((intNorthwestX >= 0) & (intNorthwestX < SIZE_3(output)) & (intNorthwestY >= 0) & (intNorthwestY < SIZE_2(output))) {
			atomicAdd(&output[OFFSET_4(output, intN, intC, intNorthwestY, intNorthwestX)], VALUE_4(input, intN, intC, intY, intX) );
		}
		if ((intNortheastX >= 0) & (intNortheastX < SIZE_3(output)) & (intNortheastY >= 0) & (intNortheastY < SIZE_2(output))) {
			atomicAdd(&output[OFFSET_4(output, intN, intC, intNortheastY, intNortheastX)], VALUE_4(input, intN, intC, intY, intX) );
		}
		if ((intSouthwestX >= 0) & (intSouthwestX < SIZE_3(output)) & (intSouthwestY >= 0) & (intSouthwestY < SIZE_2(output))) {
			atomicAdd(&output[OFFSET_4(output, intN, intC, intSouthwestY, intSouthwestX)], VALUE_4(input, intN, intC, intY, intX) );
		}
		if ((intSoutheastX >= 0) & (intSoutheastX < SIZE_3(output)) & (intSoutheastY >= 0) & (intSoutheastY < SIZE_2(output))) {
			atomicAdd(&output[OFFSET_4(output, intN, intC, intSoutheastY, intSoutheastX)], VALUE_4(input, intN, intC, intY, intX) );
		}
	} }
'''



def cupy_kernel(strFunction, objVariables):
    strKernel = globals()[strFunction]

    while True:
        objMatch = re.search('(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)

        if objMatch is None:
            break
        # end

        intArg = int(objMatch.group(2))

        strTensor = objMatch.group(4)
        intSizes = objVariables[strTensor].size()

        strKernel = strKernel.replace(objMatch.group(), str(intSizes[intArg]))
    # end

    while True:
        objMatch = re.search('(OFFSET_)([0-4])(\()([^\)]+)(\))', strKernel)

        if objMatch is None:
            break
        # end

        intArgs = int(objMatch.group(2))
        strArgs = objMatch.group(4).split(',')

        strTensor = strArgs[0]
        intStrides = objVariables[strTensor].stride()
        strIndex = [ '((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg]) + ')' for intArg in range(intArgs) ]

        strKernel = strKernel.replace(objMatch.group(0), '(' + str.join('+', strIndex) + ')')
    # end

    while True:
        objMatch = re.search('(VALUE_)([0-4])(\()([^\)]+)(\))', strKernel)

        if objMatch is None:
            break
        # end

        intArgs = int(objMatch.group(2))
        strArgs = objMatch.group(4).split(',')

        strTensor = strArgs[0]
        intStrides = objVariables[strTensor].stride()
        strIndex = [ '((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg]) + ')' for intArg in range(intArgs) ]

        strKernel = strKernel.replace(objMatch.group(0), strTensor + '[' + str.join('+', strIndex) + ']')
    # end

    return strKernel
# end

@cupy.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
    # return cupy.cuda.compile_with_cache(strKernel).get_function(strFunction)
    return cupy.RawModule(code=strKernel).get_function(strFunction)
# end

class _FunctionSoftsplat(torch.autograd.Function):
    @staticmethod
    def forward(self, input, flow):
        intSamples = input.shape[0]
        intInputDepth, intInputHeight, intInputWidth = input.shape[1], input.shape[2], input.shape[3]
        intFlowDepth, intFlowHeight, intFlowWidth = flow.shape[1], flow.shape[2], flow.shape[3]

        assert(intFlowDepth == 2)
        assert(intInputHeight == intFlowHeight)
        assert(intInputWidth == intFlowWidth)

        input = input.contiguous(); assert(input.is_cuda == True)
        flow = flow.contiguous(); assert(flow.is_cuda == True)

        output = input.new_zeros([ intSamples, intInputDepth, intInputHeight, intInputWidth ])

        if input.is_cuda == True:
            n = output.nelement()
            kernel = 'kernel_Softsplat_updateOutput' 
            cupy_launch(kernel, cupy_kernel(kernel, {
                'input': input,
                'flow': flow,
                'output': output
            }))(
                grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
                block=tuple([ 512, 1, 1 ]),
                args=[ cupy.int32(n), input.data_ptr(), flow.data_ptr(), output.data_ptr() ],
                stream=collections.namedtuple('Stream', 'ptr')(torch.cuda.current_stream().cuda_stream)
            )

        elif input.is_cuda == False:
            raise NotImplementedError()

        # end

        self.save_for_backward(input, flow)

        return output
    # end

    @staticmethod
    def backward(self, gradOutput):
        return None, None
    # end
# end

def FunctionSoftsplat(tenInput, tenFlow): 
    tenCount = _FunctionSoftsplat.apply(tenInput.new_ones(tenInput.shape[0], 1, tenInput.shape[2], tenInput.shape[3]), tenFlow).detach()
    return tenCount
# end

class Softsplat_Count(nn.Module):
    def __init__(self):
        super(Softsplat_Count, self).__init__()
    # end

    def forward(self, img, flow):
        return FunctionSoftsplat(img, flow)
    # end
# end