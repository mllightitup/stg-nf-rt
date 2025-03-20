import torch


class InferenceSTG:
    def __init__(self, args, model):
        self.model = model
        self.args = args

    def load_checkpoint(self, filename):
        try:
            checkpoint = torch.load(filename, weights_only=False)
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)
            self.model.set_actnorm_init()
            print(f"Чекпоинт загружен успешно: '{filename}'")
        except FileNotFoundError:
            print(f"No checkpoint exists from '{self.args.ckpt_dir}'. Skipping...\n")

    def test_real_time(self, data, conf):
        score = conf.amin(dim=-1)
        samp = data[0] if self.args.model_confidence else data[:, :2]
        with torch.no_grad():
            label = torch.ones(data.shape[0], device=self.args.device)
            _, nll = self.model(samp.float(), label=label, score=score)

        if self.args.model_confidence:
            nll *= score
        probs = -nll
        return probs.cpu().detach().numpy().squeeze().copy(order="C")
