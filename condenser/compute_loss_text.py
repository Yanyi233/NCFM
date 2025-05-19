import torch


def compute_match_loss(
    args,
    loader_real,
    sample_fn,
    aug_fn,
    inner_loss_fn,
    optim,
    class_list,
    timing_tracker,
    model_interval,
    data_grad,
    optim_sampling_net = None,
    sampling_net =None
):

    loss_total = 0
    match_grad_mean = 0

    # 这里是按类别进行的，class wise
    for c in class_list:
        timing_tracker.start_step()

        batch_real, _ = loader_real.class_sample(c)
        timing_tracker.record("data")
        batch_syn, _ = sample_fn(c)
        
        # 此处调用NCFM.py中的match_loss/mutil_layer_match_loss函数，对batch_real和batch_syn的batch大小没有要求，不需要匹配
        # TODO: 感觉后续可以探讨一下这部分中，两个batch_size对结果的影响
        loss = inner_loss_fn(batch_real, batch_syn, model_interval, args)
        loss_total += loss.item()
        timing_tracker.record("loss")

        optim.zero_grad()
        if optim_sampling_net is not None:
            optim_sampling_net.zero_grad()
            loss.backward(retain_graph=True)
            optim.step()
            optim.zero_grad()
            (-loss).backward()
            optim_sampling_net.step()
            optim_sampling_net.zero_grad()
        else:
            loss.backward()
            optim.step()
        if data_grad is not None:
            match_grad_mean += torch.norm(data_grad).item()
        timing_tracker.record("backward")

    return loss_total, match_grad_mean


def compute_calib_loss(
    sample_fn,
    aug_fn,
    inter_loss_fn,
    optim,
    iter_calib,
    class_list,
    timing_tracker,
    model_final,
    calib_weight,
    data_grad,
):

    calib_loss_total = 0
    calib_grad_norm = 0
    for i in range(0, iter_calib):
        for c in class_list:
            timing_tracker.start_step()

            batch_syn, label_syn = sample_fn(c)
            timing_tracker.record("data")

            # img_aug = aug_fn(torch.cat([img_syn]))
            # timing_tracker.record("aug")

            loss = calib_weight * inter_loss_fn(batch_syn, label_syn, model_final)
            calib_loss_total += loss.item()
            timing_tracker.record("loss")

            optim.zero_grad()
            loss.backward()
            if data_grad is not None:
                calib_grad_norm = torch.norm(data_grad).item()
            optim.step()
            timing_tracker.record("backward")

    return calib_loss_total, calib_grad_norm
