import torch
from torch.autograd import Variable
from torch.autograd.function import Function, once_differentiable
import apex._C

class FusedNorm(Function):
    """
    Normalizes (using L2 norm) a Variable across the slowest dimension.  
    Assumes the input Variable is contiguous.
    Internally, all calculations are performed in fp32, regardless
    of input/output precision.
    """

    @staticmethod
    def forward(ctx, input):
        """
        input is assumed to be contiguous.
        input may be either float or half precision. 
        The precision of output will match the precision of input.
        A float copy of the L2 norm across each slow dimension
        is also created and saved for the backward pass.
        """
        # torch.cuda.nvtx.range_push("FusedNorm.forward, input.size() = {}"
        #                            .format(input.size()))

        if not input.is_contiguous():
            raise RuntimeError("In FusedNorm.forward():  "
                               "input with size {} is not contiguous"
                               .format(input.size()))
        if not input.is_cuda:
            raise RuntimeError("In FusedNorm.forward():  input.is_cuda = False."
                               "Currently, only cuda tensors are supported.")
 
        """
        This is ok, new() treats a torch.Size object properly.
        No need to unpack with an asterisk via new(*input.size()).
        """
        output = input.new(input.size()).contiguous()

        """
        For output with size (slow, faster, faster, ...fastest), we may want
        norms with size (slow, 1, 1, ...1), so that if you want retrieve norms 
        and apply the same normalizing factors to another Tensor "t" with the 
        same size as output, "t/norms" will broadcast each element of norms 
        across the corresponding slowest dim of t.
        """
        norm_size = (output.size(0),) + (1,)*(output.dim()-1)
        norms = torch.cuda.FloatTensor(*norm_size).contiguous()
        """
        Beware:  If you call the following:
        norms = torch.cuda.FloatTensor(norm_size).contiguous()
        the constructor sees a tuple:
        FloatTensor( (output_size(0),1,1,...) )
        and creates a 1D tensor with values from the tuple:
        [output_size(0),1,1,...].
        """

        # torch.cuda.synchronize()

        # print("norms = ", norms)
        # print("norms.size  () = ", norms.size())
        # print("norms.stride() = ", norms.stride())

        # print("type(input)  = ", type(input))
        # print("type(output) = ", type(output))
        # print("type(norms)  = ", type(norms))
        # print( "input.data_ptr = {:x}".format(input.data_ptr()))
        
        apex._C.norm_fwd(input, output, norms)
        # apex._C.norm_fwd(input.data, output.data, norms)

        # torch.cuda.synchronize()

        # print("norms in forward():  ", norms)

        ctx.save_for_backward(input)

        # save_for_backward can only save input or output tensors,
        # so here's a hacky workaround to save the norms:
        ctx.norms = norms

        # torch.cuda.nvtx.range_pop()

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """
        grad_output is assumed to be contiguous.
        grad_output may be either float or half precision.
        The precision of grad_input will match the precision of grad_output.
        """
        # torch.cuda.nvtx.range_push("FusedNorm.backward, grad_output.size() = {}"
        #                            .format(grad_output.size()))

        if not grad_output.is_cuda:
            raise RuntimeError("In FusedNorm.backward():  grad_output.is_cuda = False."
                               "Currently, only cuda tensors are supported.")

        savedInput, = ctx.saved_tensors
        norms = ctx.norms

        # better safe than sorry
        grad_output_contig = grad_output.contiguous()
        grad_input = grad_output_contig.new(grad_output.size()).contiguous()

        apex._C.norm_bwd(grad_output_contig, grad_input, savedInput, norms)
        # apex._C.norm_bwd(grad_output_contig.data, grad_input.data, savedInput.data, norms)

        # torch.cuda.nvtx.range_pop()

        # print("\n\n")
        # print("grad_output.is_contiguous() = {:x}".format(grad_output.is_contiguous()))
        # print(" grad_input.is_contiguous() = {:x}".format( grad_input.is_contiguous()))
        # print(" savedInput.is_contiguous() = {:x}".format( savedInput.is_contiguous()))
        # print("      norms.is_contiguous() = {:x}".format(      norms.is_contiguous()))
        # print("\n\n")
        # print("grad_output.data_ptr = {:x}".format(grad_output.data_ptr()))
        # print(" grad_input.data_ptr = {:x}".format( grad_input.data_ptr()))
        # print(" savedInput.data_ptr = {:x}".format( savedInput.data_ptr()))
        # print("      norms.data_ptr = {:x}".format(      norms.data_ptr()))
        # print("\n\n")
        # print("grad_output in backward():  ", grad_output)
        # print(" grad_input in backward():  ", grad_input)
        # print(" savedInput in backward():  ", savedInput)
        # print("      norms in backward():  ", norms)

        return grad_input
