from metrics import get_best_threshold_accuracy, get_cp_detection_metrics, covering_metric
from sklearn.metrics import accuracy_score
from utils import find_change_points

def eval_classification_methods(result_list):
    half_results_1_list = []
    half_results_2_list = []
    for results in result_list:
        n_samples = len(results)
        half_samples = n_samples >> 1
        half_results_1 = results[:half_samples]
        half_results_2 = results[half_samples:len(results)]
        half_results_1_list.extend(half_results_1)
        half_results_2_list.extend(half_results_2)
    
    _, best_thres1 = get_best_threshold_accuracy([y for x in half_results_1 for y in x['labels']], [y for x in half_results_1 for y in x['predictions']])
    _, best_thres2 = get_best_threshold_accuracy([y for x in half_results_2 for y in x['labels']], [y for x in half_results_2 for y in x['predictions']])

    new_results_list = []
    for results in result_list:
        new_results = []
        n_samples = len(results)
        for i in range(n_samples):
            if i < half_samples:
                best_thres = best_thres2
            else:
                best_thres = best_thres1
            num_sentence = len(results[i]['labels'])
            est_label_list = [(x > best_thres).astype(int) for x in results[i]['predictions']]
            acc = accuracy_score(results[i]['labels'], est_label_list)
            true_cp, est_cp = find_change_points(results[i]['labels']), find_change_points(est_label_list)
            eval_true_cp = true_cp + [num_sentence] if len(true_cp) > 0 else [0] + [num_sentence]
            eval_est_cp = est_cp + [num_sentence] if len(est_cp) > 0 else [0] + [num_sentence]
            ri, hau = get_cp_detection_metrics(eval_true_cp, eval_est_cp)
            cover = covering_metric(results[i]['labels'], est_label_list)

            new_results.append({
                "labels": results[i]['labels'], "predictions": results[i]['predictions'], "best_thres": best_thres, "true_cp": true_cp, "est_cp": est_cp,  
                "acc": acc, "rand": ri, "hausdorff": hau, 'cp_num_diff': len(true_cp) - len(est_cp), "covering": cover})
    
        new_results_list.append(new_results)
    return new_results_list

