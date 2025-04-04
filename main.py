import torch
class CustomFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_t, t, sigma_data):
        ctx.save_for_backward(x_t, t)
        ctx.sigma_data = sigma_data
        return (x_t / sigma_data) * t

    @staticmethod
    def backward(ctx, grad_output):
        x_t, t = ctx.saved_tensors
        sigma_data = ctx.sigma_data
        grad_x_t = (t / sigma_data) * grad_output
        grad_t = (x_t / sigma_data) * grad_output
        return grad_x_t, grad_t, None

    @staticmethod
    def jvp(ctx, x_t_tangent, t_tangent):
        x_t, t = ctx.saved_tensors
        sigma_data = ctx.sigma_data
        primal_output = (x_t / sigma_data) * t
        df_dx_t = t / sigma_data
        df_dt = x_t / sigma_data
        tangent_output = df_dx_t * x_t_tangent + df_dt * t_tangent
        return primal_output, tangent_output

# 使用示例
x_t = torch.tensor([4.0], requires_grad=True)
t = torch.tensor([2.0], requires_grad=True)
v_x_t = torch.tensor([1.0])
v_t = torch.tensor([1.0])

# 启用前向模式并调用
with torch.autograd.forward_ad.dual_level():
    dual_x_t = torch.autograd.forward_ad.make_dual(x_t, v_x_t)
    dual_t = torch.autograd.forward_ad.make_dual(t, v_t)
    output = CustomFunction.apply(dual_x_t, dual_t, 2.0)
    primal, tangent = torch.autograd.forward_ad.unpack_dual(output)

print("Primal:", primal)  # 输出: tensor([4.])
print("Tangent:", tangent)  # 输出: tensor([3.])