import torch
import unittest
from models.scm_model.wan_scm import WanModelSCM

class TestWanModelSCM(unittest.TestCase):
    def setUp(self):
        # 初始化模型
        self.hidden_size = 64
        self.model = WanModelSCM(
            hidden_size=self.hidden_size,
            logvar=True,
            logvar_scale_factor=1.0
        )
        self.model.eval()

    def test_trigflow_transformation(self):
        # 测试TrigFlow变换
        batch_size = 2
        channels = 3
        frames = 4
        height = 8
        width = 8

        # 创建输入数据
        x = [torch.randn(channels, frames, height, width) for _ in range(batch_size)]
        timestep = torch.tensor([0.3, 0.7])  # 在[0, np.pi/2]范围内
        context = [torch.randn(5, 32) for _ in range(batch_size)]  # 文本嵌入
        seq_len = 10

        # 前向传播
        output = self.model(x, timestep, context, seq_len)

        # 验证输出形状
        self.assertEqual(len(output), batch_size)
        self.assertEqual(output[0].shape, (channels, frames, height//8, width//8))

        # 验证TrigFlow变换的数值范围
        for out in output:
            self.assertTrue(torch.isfinite(out).all())

    def test_jvp_computation(self):
        # 测试JVP计算
        batch_size = 2
        channels = 3
        frames = 4
        height = 8
        width = 8

        x = [torch.randn(channels, frames, height, width) for _ in range(batch_size)]
        timestep = torch.tensor([0.3, 0.7])
        context = [torch.randn(5, 32) for _ in range(batch_size)]
        seq_len = 10

        # 使用jvp=True进行前向传播
        output_with_jvp = self.model(x, timestep, context, seq_len, jvp=True)

        # 验证JVP输出
        self.assertEqual(len(output_with_jvp), batch_size)
        for out in output_with_jvp:
            self.assertTrue(torch.isfinite(out).all())

    def test_logvar_return(self):
        # 测试logvar返回功能
        batch_size = 2
        channels = 3
        frames = 4
        height = 8
        width = 8

        x = [torch.randn(channels, frames, height, width) for _ in range(batch_size)]
        timestep = torch.tensor([0.3, 0.7])
        context = [torch.randn(5, 32) for _ in range(batch_size)]
        seq_len = 10

        # 使用return_logvar=True进行前向传播
        output, logvar = self.model(x, timestep, context, seq_len, return_logvar=True)

        # 验证输出和logvar
        self.assertEqual(len(output), batch_size)
        self.assertIsInstance(logvar, torch.Tensor)
        self.assertEqual(logvar.shape, (batch_size, 1))
        self.assertTrue(torch.isfinite(logvar).all())

if __name__ == '__main__':
    unittest.main()