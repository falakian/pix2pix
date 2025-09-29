import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import List, Tuple
import torchvision.transforms as transforms

from kraken import rpred, pageseg, binarization
from kraken.lib import models

def print_torch_state(tag):
    print(f"--- {tag} ---")
    print("cudnn.enabled", torch.backends.cudnn.enabled)
    print("cudnn.benchmark", torch.backends.cudnn.benchmark)
    print("cudnn.deterministic", torch.backends.cudnn.deterministic)
    print("deterministic_algorithms",
          torch.are_deterministic_algorithms_enabled()
          if hasattr(torch, "are_deterministic_algorithms_enabled") else None)
    print("threads", torch.get_num_threads(), "interop", torch.get_num_interop_threads())


class KrakenOCRWrapper(nn.Module):
    """
    Kraken OCR wrapper .
    - Loads a .mlmodel
    - Provides differentiable forward (logits) for generator outputs
    - Extracts labels from real images using OCR
    - Encodes labels with Kraken codec for CTC
    """

    def __init__(self, model_path: str, device: str = 'cuda'):
        super().__init__()
        self.device = device

        # Load Kraken model
        self.model = models.load_any(model_path, device='cpu')
        torch.set_grad_enabled(True)
        self.net = copy.deepcopy(self.model.nn.nn).to(device)
        self.net.train()
        for p in self.net.parameters():
            p.requires_grad = False  # Freeze parameters

        self.codec = self.model.codec
        blank_idx = self.codec.c2l.get(' ', 1)
        self.ctc = nn.CTCLoss(blank=blank_idx[0], zero_infinity=True)
        self.input_width = 960

    @torch.no_grad()
    def ocr_text(self, img_tensor: torch.Tensor) -> List[str]:
        """
        Runs OCR on real target images (B,C,H,W) and returns recognized strings.
        """
        out_texts = []
        for b in range(img_tensor.shape[0]):
            pil = self._to_pil(img_tensor[b:b+1])
            seg = pageseg.segment(pil)
            recognizer = rpred.rpred(network=self.model, im=pil, bounds=seg)
            txt = ''.join([rec.prediction for rec in recognizer])
            out_texts.append(txt)
        return out_texts

    def forward_logits(self, img_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable forward pass for generator outputs.
        Returns (logits, logit_lengths) suitable for CTC loss.
        """
        self.net.train()
        x = self._prep_for_kraken(img_tensor)  # (B,1,H,W)
        output_tuple = self.net(x)  # shape (B,C,T) or (B,C,H,W)
        y = output_tuple[0]
        if y.dim() == 3:
            y = y.permute(2, 0, 1).contiguous()  # (T,B,C)
        elif y.dim() == 4:
            B, C, H, W = y.shape
            y = y.squeeze(2).permute(2, 0, 1).contiguous()  # (T,B,C)
        else:
            raise RuntimeError("Unexpected Kraken logits shape")

        T = y.shape[0]
        logit_lengths = torch.full((y.shape[1],), T, dtype=torch.long, device=y.device)
        return y, logit_lengths

    def encode_texts(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes strings into targets + lengths for CTCLoss.
        """
        targets = []
        lengths = []
        for t in texts:
            enc = self.codec.encode(t)
            targets.extend(enc)
            lengths.append(len(enc))
        if len(targets) == 0:
            targets = self.codec.c2l.get(' ', 1)
            lengths = [1]
        targets = torch.tensor(targets, dtype=torch.long, device=self.device)
        lengths = torch.tensor(lengths, dtype=torch.long, device=self.device)
        return targets, lengths

    def ctc_loss(self, logits, logit_lengths, targets, target_lengths) -> torch.Tensor:
        """
        Computes CTC loss for given logits and targets.
        """
        log_probs = F.log_softmax(logits, dim=-1)
        return self.ctc(log_probs, targets, logit_lengths, target_lengths)

    def _to_pil(self, x: torch.Tensor) -> Image.Image:
        to_pil = transforms.ToPILImage()
        x = x[0, 0, :, :] 
        x = (x + 1.0) / 2.0
        img = to_pil(x.clamp(0, 1))
        bw_img = binarization.nlbin(img)
        return bw_img

    def _prep_for_kraken(self, x: torch.Tensor) -> torch.Tensor:
        x = x[0:1,0:1, :, :] 
        x = (x + 1.0) / 2.0
        B,_ ,H, W = x.shape
        newW = self.input_width
        newH = (H * newW) // W
        img = F.interpolate(x, size=(newH, newW), mode='bilinear', align_corners=False)
        return img