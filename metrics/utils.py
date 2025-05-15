import numpy as np

from sklearn import metrics


def parse_metric(metric_dict):
    if metric_dict is None:
        return "\n"
    str = "\n"
    str += "================================ Each dataset best metric ================================ \n"
    for key, value in metric_dict.items():
        if key != 'avg':
            str= str+ f"| {key}: "
            for k,v in value.items():
                str = str + f" {k}={v} "
            str= str+ "| \n"
        else:
            str += "============================================================================================= \n"
            str += "================================== Average best metric ====================================== \n"
            avg_dict = value
            for avg_key, avg_value in avg_dict.items():
                if avg_key == 'dataset_dict':
                    for key,value in avg_value.items():
                        str = str + f"| {key}: {value} | \n"
                else:
                    str = str + f"| avg {avg_key}: {avg_value} | \n"
    str += "============================================================================================="
    return str


def get_test_metrics(y_pred,
                     y_true,
                     # img_names,
                     ):
    def get_video_metrics(image, pred, label):
        result_dict = {}
        new_label = []
        new_pred = []
        # print(image[0])
        # print(pred.shape)
        # print(label.shape)
        for item in np.transpose(np.stack((image, pred, label)), (1, 0)):

            s = item[0]
            if '\\' in s:
                parts = s.split('\\')
            else:
                parts = s.split('/')
            a = parts[-2]
            b = parts[-1]

            if a not in result_dict:
                result_dict[a] = []

            result_dict[a].append(item)
        image_arr = list(result_dict.values())

        for video in image_arr:
            pred_sum = 0
            label_sum = 0
            leng = 0
            for frame in video:
                pred_sum += float(frame[1])
                label_sum += int(frame[2])
                leng += 1
            new_pred.append(pred_sum / leng)
            new_label.append(int(label_sum / leng))
        fpr, tpr, thresholds = metrics.roc_curve(new_label, new_pred)
        v_auc = metrics.auc(fpr, tpr)
        fnr = 1 - tpr
        v_eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        return v_auc, v_eer


    y_pred = y_pred.squeeze()
    # For UCF, where labels for different manipulations are not consistent.
    y_true[y_true >= 1] = 1
    # auc
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    # eer
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    # ap
    ap = metrics.average_precision_score(y_true, y_pred)
    # acc
    prediction_class = (y_pred > 0.5).astype(int)
    correct = (prediction_class == np.clip(y_true, a_min=0, a_max=1)).sum().item()
    acc = correct / len(prediction_class)

    # mid = 0.5 if np.min(y_pred) >= 0 and np.max(y_pred) <= 1 else (np.min(y_pred) + np.max(y_pred)) / 2
    # threshold_candidates = np.sort(np.unique(np.concatenate(
    #     [np.array([mid]),
    #      y_pred,
    #      np.linspace(np.min(y_pred), np.max(y_pred), 100)]  # add more threshold candidates
    # )))
    # best_threshold, best_accuracy = 0, 0
    # for threshold_candidate in threshold_candidates:
    #     pred_labels = (y_pred >= threshold_candidate).astype(int)
    #     accuracy = metrics.accuracy_score(y_true, pred_labels)
    #     if accuracy > best_accuracy:
    #         best_threshold = threshold_candidate
    #         best_accuracy = accuracy
    #     elif accuracy == best_accuracy:  # choose threshold closed to (min + max) / 2, i.e., 0.5 for [0, 1] case
    #         if abs(threshold_candidate - mid) < abs(best_threshold - mid):
    #             best_threshold = threshold_candidate
    # print(f'Optimal accuracy threshold: {best_threshold:.6f}')

    # if type(img_names[0]) is not list:
    #     # calculate video-level auc for the frame-level methods.
    #     v_auc, _ = get_video_metrics(img_names, y_pred, y_true)
    # else:
    #     # video-level methods
    #     v_auc=auc
    v_auc = auc

    return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap, 'pred': y_pred, 'video_auc': v_auc, 'label': y_true}
    # return {'acc': best_accuracy, 'auc': auc, 'eer': eer, 'ap': ap, 'pred': y_pred, 'video_auc': v_auc, 'label': y_true}
