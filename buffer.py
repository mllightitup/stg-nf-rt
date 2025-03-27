import torch


class PersonBuffer:
    def __init__(self, max_history, num_keypoints, device, dims=3):
        self.kps = torch.zeros((max_history, num_keypoints, dims), device=device)
        self.conf = torch.zeros(max_history, device=device)
        self.ptr = 0
        self.count = 0
        self.max_history = max_history

    def add(self, new_kp, new_conf):
        self.kps[self.ptr] = new_kp
        self.conf[self.ptr] = new_conf
        self.ptr = (self.ptr + 1) % self.max_history
        if self.count < self.max_history:
            self.count += 1

    def is_full(self):
        return self.count == self.max_history

    def get_ordered_history(self):
        if self.ptr == 0:
            return self.kps, self.conf
        kps_ordered = torch.cat([self.kps[self.ptr :], self.kps[: self.ptr]], dim=0)
        conf_ordered = torch.cat([self.conf[self.ptr :], self.conf[: self.ptr]], dim=0)
        return kps_ordered, conf_ordered


class BufferManager:
    def __init__(self, max_history, device):
        self.max_history = max_history
        self.device = device
        self.track_histories = {}  # track_id -> TrackHistory
        self.tmp_histories = {}
        self.frame_counter = 0
        self.built_tensors = ()

    def update(self, tids, kps, confs):
        """
        tids: (N,) track IDs
        kps: (N, num_keypoints, 3)
        confs: (N,) confidence значений bbox или любых других
        """
        self.tmp_histories = {} # буфер для тех, кого видим в текущем кадре
        self.built_tensors = (None, None, None)
        valid_kps = []
        valid_conf = []
        valid_ids = []
        tid_list = tids.tolist()
        for tid, kp, c in zip(tid_list, kps, confs):
            if tid not in self.track_histories:
                # num_keypoints, dims
                num_kp, dims = kp.shape
                self.tmp_histories[tid] = PersonBuffer(
                    self.max_history, num_kp, self.device, dims
                )
            else:
               self.tmp_histories[tid] =  self.track_histories[tid]
            self.tmp_histories[tid].add(kp, c)
            
            if self.tmp_histories[tid].is_full():  # собираем полные патчи кадров для built_tensors
                kps_ord, conf_ord = self.tmp_histories[tid].get_ordered_history()
                valid_kps.append(kps_ord)
                valid_conf.append(conf_ord)
                valid_ids.append(tid)

        if valid_kps:
            tensor_kps = torch.stack(valid_kps, dim=0)  # (M, max_history, num_keypoints, 3)
            tensor_conf = torch.stack(valid_conf, dim=0)  # (M, max_history)
            union_ids = torch.tensor(valid_ids, dtype=torch.int64, device=self.device)
            self.built_tensors = (tensor_kps, tensor_conf, union_ids)

        # Удаляем из трекера тех, кого не видим в текущем кадре
        self.track_histories = self.tmp_histories
    
    def build_tensor_deprecated(self):
        """
        Возвращает:
          - (M, max_history, num_keypoints, 3) keypoints
          - (M, max_history) conf
          - (M,) track_ids
        Только для тех track_id, где история "полная".
        """
        valid_kps = []
        valid_conf = []
        valid_ids = []
        for tid, hist in self.track_histories.items():
            if hist.is_full():
                kps_ord, conf_ord = hist.get_ordered_history()
                valid_kps.append(kps_ord)
                valid_conf.append(conf_ord)
                valid_ids.append(tid)

        if not valid_kps:
            return None, None, None
        tensor_kps = torch.stack(valid_kps, dim=0)  # (M, max_history, num_keypoints, 3)
        tensor_conf = torch.stack(valid_conf, dim=0)  # (M, max_history)
        union_ids = torch.tensor(valid_ids, dtype=torch.int64, device=self.device)

        return tensor_kps, tensor_conf, union_ids
