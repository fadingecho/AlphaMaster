"""Regression tests for the 20-stock neutral-zero normalization."""
import unittest
import torch
from model_core.vm import StackVM
from config import Config
from strategy_manager.signal import compute_target_positions_stateless


class NeutralZeroTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.normalize=StackVM._normalize_output

    def test_constant_days_inside_nonconstant_history(self):
        for dtype in [torch.float32,torch.float64]:
            for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
                x=torch.arange(20*1178,device=device,dtype=dtype).reshape(20,1178)/100
                for i,value in enumerate([0.,-.6931471824645996,1.,-4.5]):
                    x[:,800+i]=value
                y=self.normalize(x)
                self.assertTrue(torch.equal(y[:,800:804],torch.zeros_like(y[:,800:804])))
                self.assertTrue(torch.isfinite(y).all())
                self.assertEqual(y.dtype,x.dtype)
                self.assertEqual(y.device,x.device)

    def test_all_history_constant_is_neutral(self):
        for value in [0.,-.6931471824645996,3.]:
            x=torch.full((20,1178),value)
            self.assertEqual(torch.count_nonzero(self.normalize(x)).item(),0)

    def test_near_constant_roundoff_is_neutral(self):
        x=torch.full((20,50),-.6931471824645996)
        x[0,:]=torch.nextafter(x[0,:],torch.zeros_like(x[0,:]))
        self.assertEqual(torch.count_nonzero(self.normalize(x)).item(),0)

    def test_meaningful_spread_matches_float64_reference(self):
        x=torch.linspace(-2,3,20).reshape(20,1).repeat(1,40)
        expected=((x.double()-x.double().mean(0))/x.double().std(0)).clamp(-3,3).float()
        torch.testing.assert_close(self.normalize(x),expected,rtol=1e-6,atol=1e-7)
        # Small units do not suppress a real cross-sectional difference.
        torch.testing.assert_close(self.normalize(x*1e-10),expected,rtol=1e-6,atol=1e-7)

    def test_prefix_invariance_and_noncontiguous_layout(self):
        g=torch.Generator().manual_seed(29)
        x=torch.randn(20,1178,generator=g)
        x[:,300:360]=-.6931471824645996
        full=self.normalize(x)
        for end in [1,19,300,301,359,360,499,500,501,900,1177]:
            for prefix in [x[:,:end],x[:,:end].clone(),x[:,:end].T.contiguous().T]:
                torch.testing.assert_close(self.normalize(prefix),full[:,:end],rtol=1e-6,atol=1e-7)
        # The former whole-history early return must not affect old columns.
        constant=torch.full((20,40),-.6931471824645996)
        extended=torch.cat([constant,x],dim=1)
        self.assertTrue(torch.equal(self.normalize(constant),self.normalize(extended)[:,:40]))

    def test_zero_factors_map_to_zero_long_only_positions(self):
        previous = Config.A_SHARE_LONG_ONLY
        self.addCleanup(setattr, Config, 'A_SHARE_LONG_ONLY', previous)
        Config.A_SHARE_LONG_ONLY = True
        y=self.normalize(torch.full((20,1178),-.6931471824645996))
        self.assertEqual(torch.count_nonzero(compute_target_positions_stateless(y)).item(),0)


if __name__=='__main__':
    unittest.main(verbosity=2)
