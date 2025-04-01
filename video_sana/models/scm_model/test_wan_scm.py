import torch
import unittest
import sys
import os
import math
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.scm_model.wan_scm import WanModelSCM

class TestWanModelSCM(unittest.TestCase):
    def setUp(self):
        # 初始化模型
        self.model = WanModelSCM(
            logvar=True,
            logvar_scale_factor=1.0
        )
        self.model.to('cuda')

        self.model.eval()

    def test_trigflow_transformation(self):
        # 测试TrigFlow变换
        batch_size = 1
        channels = 16
        frames = 21
        height = 30
        width = 52

        # 创建输入数据
        x = [torch.randn(channels, frames, height, width).cuda() for _ in range(batch_size)]
        timestep = torch.tensor([0.3]).cuda()
        context = [torch.randn(512, 4096).cuda() for _ in range(batch_size)]  # 文本嵌入
        target_shape = (16, 21, 60, 104)
        patch_size = (1, 2, 2)
        sp_size = 1
        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                          (patch_size[1] * patch_size[2]) *
                          target_shape[1] / sp_size) * sp_size

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
        batch_size = 1
        channels = 16
        frames = 21
        height = 30
        width = 52

        x = [torch.randn(channels, frames, height, width).cuda() for _ in range(batch_size)]
        timestep = torch.tensor([0.3]).cuda()
        context = [torch.randn(512, 4096).cuda() for _ in range(batch_size)]
        target_shape = (16, 21, 60, 104)
        patch_size = (1, 2, 2)
        sp_size = 1
        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                          (patch_size[1] * patch_size[2]) *
                          target_shape[1] / sp_size) * sp_size

        # 使用jvp=True进行前向传播
        output_with_jvp = self.model(x, timestep, context, seq_len, jvp=True)

        # 验证JVP输出
        self.assertEqual(len(output_with_jvp), batch_size)
        for out in output_with_jvp:
            self.assertTrue(torch.isfinite(out).all())

    def test_logvar_return(self):
        # 测试logvar返回功能
        batch_size = 1
        channels = 16
        frames = 21
        height = 30
        width = 52

        x = [torch.randn(channels, frames, height, width).cuda() for _ in range(batch_size)]
        timestep = torch.tensor([0.3]).cuda()
        context = [torch.randn(512, 4096).cuda() for _ in range(batch_size)]
        target_shape = (16, 21, 60, 104)
        patch_size = (1, 2, 2)
        sp_size = 1
        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                          (patch_size[1] * patch_size[2]) *
                          target_shape[1] / sp_size) * sp_size

        # 使用return_logvar=True进行前向传播
        output, logvar = self.model(x, timestep, context, seq_len, return_logvar=True)

        # 验证输出和logvar
        self.assertEqual(len(output), batch_size)
        self.assertIsInstance(logvar, torch.Tensor)
        self.assertEqual(logvar.shape, (batch_size, 1))
        self.assertTrue(torch.isfinite(logvar).all())

if __name__ == '__main__':
    unittest.main()