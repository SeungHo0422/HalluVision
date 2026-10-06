import ujson
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# 파일 경로 (gpt-4o-mini는 summary_ner, 3.5-turbo는 cyner_sum_results_250427)
original_path = "original_ner.json"
summary_path_4o = "summary_ner.json"
summary_path_35 = "cyner_sum_results_250427.json"

# JSON 로드
with open(original_path, "r") as f:
    original_data = ujson.load(f)
with open(summary_path_4o, "r") as f:
    summary_data_4o = ujson.load(f)
with open(summary_path_35, "r") as f:
    summary_data_35 = ujson.load(f)

# NER 레이블 종류
LABELS = ["Malware", "System", "Indicator", "Vulnerability", "Organization"]
LABELS_DISPLAY = ['MAL', 'SYS', 'IND', 'VUL', 'ORG']

# 전체 통계 저장
label_freq_stats_4o = {label: {'y_true': 0, 'y_pred': 0} for label in LABELS}
label_stats_4o = {label: {'y_true': [], 'y_pred': []} for label in LABELS}
label_freq_stats_35 = {label: {'y_true': 0, 'y_pred': 0} for label in LABELS}
label_stats_35 = {label: {'y_true': [], 'y_pred': []} for label in LABELS}
doc_stats_4o = {}
doc_stats_35 = {}
loss_vectors_4o = []
loss_vectors_35 = []
false_positives_4o = {}
false_positives_35 = {}


def evaluate(summary_path, summary_data, label_stats, label_freq_stats, doc_stats, false_positives, loss_vectors):
    for doc_id in summary_data:
        orig = original_data[doc_id]
        summ = summary_data[doc_id]


        # doc_id 문서에서의 true, pred
        doc_y_true = []
        doc_y_pred = []

        for label in LABELS:
            true_entities = set(orig.get(label, []))

            if summary_path == "summary_ner_3.5.json":
                preprocessed_pred_entities = set(summ.get(label, []))
                pred_entities = set()
                for entity in preprocessed_pred_entities:
                    pred_entities.update(set(map(lambda x: x.strip(), entity.split(','))))
            else:
                pred_entities = set(summ.get(label, []))

            # False Positives 저장
            fp_entities = pred_entities - true_entities
            if fp_entities:
                if doc_id not in false_positives:
                    false_positives[doc_id] = {}
                false_positives[doc_id][label] = list(fp_entities)

            # for ent in pred_entities:
            #     label_stats[label]['y_pred'].append(1)
            #     label_stats[label]['y_true'].append(1 if ent in true_entities else 0)
            #     label_freq_stats[label]['y_pred'] += 1
            #     label_freq_stats[label]['y_true'] += 1 if ent in true_entities else 0
            #     doc_y_pred.append(1)
            #     doc_y_true.append(1 if ent in true_entities else 0)
            #
            # for ent in true_entities - pred_entities:
            #     label_stats[label]['y_pred'].append(0)
            #     label_stats[label]['y_true'].append(1)
            #     label_freq_stats[label]['y_pred'] += 1
            #     label_freq_stats[label]['y_true'] += 1 if ent in true_entities else 0
            #     doc_y_pred.append(0)
            #     doc_y_true.append(1)

            for true in true_entities:
                label_stats[label]['y_true'].append(1)
                label_stats[label]['y_pred'].append(1 if true in pred_entities else 0)
                label_freq_stats[label]['y_true'] += 1
                label_freq_stats[label]['y_pred'] += 1 if true in pred_entities else 0
                doc_y_true.append(1)
                doc_y_pred.append(1 if true in pred_entities else 0)

            for pred_item in pred_entities:
                if pred_item not in true_entities:
                    label_stats[label]['y_true'].append(0)
                    label_stats[label]['y_pred'].append(1)
                    label_freq_stats[label]['y_true'] += 1
                    label_freq_stats[label]['y_pred'] += 1 if true in pred_entities else 0
                    doc_y_true.append(0)
                    doc_y_pred.append(1)


        # 모든 문서에서 loss 계산
        orig_counts = np.array([len(orig.get(label, [])) for label in LABELS])
        summ_counts = np.array([len(summ.get(label, [])) for label in LABELS])

        # 정규화
        orig_total = orig_counts.sum()
        summ_total = summ_counts.sum()

        orig_ratio = orig_counts / orig_total if orig_total != 0 else np.zeros(len(LABELS))
        summ_ratio = summ_counts / summ_total if summ_total != 0 else np.zeros(len(LABELS))

        loss = np.abs(orig_ratio - summ_ratio)
        loss_vectors.append(loss)

        # per-document precision/recall/f1. A document with no entity on either
        # side has nothing to score; scikit-learn >= 1.6 raises on empty input
        # where older versions returned zeros, so keep the legacy value.
        if doc_y_true:
            precision, recall, f1, _ = precision_recall_fscore_support(
                doc_y_true, doc_y_pred, average='binary', zero_division=0
            )
        else:
            precision = recall = f1 = 0.0
        doc_stats[doc_id] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    loss_matrix = np.array(loss_vectors) #shape = (4940, 5)

    # FP 저장
    if summary_path == summary_path_35:
        fp_savepath = "false_positives_gpt35.json"
        modelname = "gpt-3.5-turbo"
    elif summary_path == summary_path_4o:
        fp_savepath = "false_positives_gpt4o.json"
        modelname = "gpt-4o-mini"

    with open(fp_savepath, "w") as f:
        ujson.dump(false_positives, f, indent=4)

    # 해당 모델명 출력
    print()
    print("-"*15)
    print(f"\nEvaluate Dataset Path : {modelname}")

    # label별 정밀도, 재현율, F1 출력
    print("Label-wise Metrics:")
    total_true = []
    total_pred = []
    for label in LABELS:
        y_true = label_stats[label]['y_true']
        y_pred = label_stats[label]['y_pred']
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fdr = fp / (fp + tp)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        print(f"🔹 {label}")
        print(f"   Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, FDR: {fdr:.4f}")
        total_true.extend(y_true)
        total_pred.extend(y_pred)

    t_prec, t_recall, t_f1, _ = precision_recall_fscore_support(
        total_true, total_pred, average='binary', zero_division=0
    )
    total_tn, total_fp, total_fn, total_tp = confusion_matrix(total_true, total_pred).ravel()
    total_fdr = total_fp / (total_fp + total_tp)
    print(f"Total Precision: {t_prec:.4f}, Total recall: {t_recall:.4f}, Total F1: {t_f1:.4f}, Total FDR: {total_fdr:.4f}")

    print("\nLabel Frequency")
    for label in LABELS:
        y_true = label_freq_stats[label]['y_true']
        y_pred = label_freq_stats[label]['y_pred']

        print(f"{label} - 원문: {y_true}개, 요약문: {y_pred}개")


def init_dataframe_docs(doc_stats):
    df_doc = pd.DataFrame.from_dict(doc_stats, orient='index')
    df_doc.index.name = 'doc_id'
    df_doc.sort_index(inplace=True)
    return df_doc

"""시각화 (CDF of prec, recall, f1)"""
def cdf_prec_recall(df_doc_4o, df_doc_35):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))  # 논문용 가로형 작게

    model_labels = ['gpt-4o-mini', 'gpt-3.5-turbo']
    model_colors = {
        'gpt-4o-mini': 'green',
        'gpt-3.5-turbo': 'orange'
    }

    for i, metric in enumerate(['precision', 'recall']):
        ax = axes[i]
        for df_doc, model_name in zip([df_doc_4o, df_doc_35], model_labels):
            values = df_doc[metric].dropna().values
            sorted_vals = np.sort(values)
            cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            ax.plot(sorted_vals, cdf, label=model_name, color=model_colors[model_name])

        ax.set_xlabel("Score", fontsize=9)
        if i == 0:
            ax.set_ylabel("Cumulative Probability", fontsize=9)
        ax.set_title(metric.capitalize(), fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=8, loc='lower right', title='Model', title_fontsize=8)

    plt.tight_layout()
    plt.savefig("figures/figure5.pdf", dpi=300)
    plt.show()


"""시각화 (Frequency-based Entity Similarity)"""
def freq_based_entity_similarity(label_freq_stats_4o, label_freq_stats_35):
    # gpt-4o-mini
    true_counts_4o = [label_freq_stats_4o[l]["y_true"] for l in LABELS]
    pred_counts_4o = [label_freq_stats_4o[l]["y_pred"] for l in LABELS]
    true_pct_4o = [x / sum(true_counts_4o) * 100 for x in true_counts_4o]
    pred_pct_4o = [x / sum(pred_counts_4o) * 100 for x in pred_counts_4o]

    # gpt-3.5-turbo
    true_counts_35 = [label_freq_stats_35[l]["y_true"] for l in LABELS]
    pred_counts_35 = [label_freq_stats_35[l]["y_pred"] for l in LABELS]
    true_pct_35 = [x / sum(true_counts_35) * 100 for x in true_counts_35]
    pred_pct_35 = [x / sum(pred_counts_35) * 100 for x in pred_counts_35]

    # Subplots
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)

    for ax, model_name, true_pct, pred_pct in zip(
            axes,
            ['gpt-4o-mini', 'gpt-3.5-turbo'],
            [true_pct_4o, true_pct_35],
            [pred_pct_4o, pred_pct_35]
    ):
        ax.plot(LABELS_DISPLAY, true_pct, marker='o', linestyle='-', label='Original')
        ax.plot(LABELS_DISPLAY, pred_pct, marker='s', linestyle='--', label='Summary')
        ax.set_title(model_name, fontsize=10)
        ax.set_xlabel("Entity Label", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.5)

    axes[0].set_ylabel("Entity Frequency (%)", fontsize=9)

    plt.tight_layout()
    plt.savefig('figures/figure6.pdf', dpi=300)
    plt.show()

# """시각화 (각 문서별 loss Hitmap)"""
# plt.figure(figsize=(12, 6))
# sns.heatmap(loss_matrix.T, cmap="YlOrRd", xticklabels=False, yticklabels=LABELS, cbar_kws={'label': 'Loss'})
# plt.title("Per-Document Label-wise Distribution Loss (Summary vs Original)")
# plt.xlabel("Document Index")
# plt.ylabel("Label")
# plt.tight_layout()
# plt.show()
#
# df_loss = pd.DataFrame(loss_matrix, columns=LABELS)
# """시각화 (각 문서별 Boxplot)"""
# plt.figure(figsize=(10, 6))
# sns.boxplot(data=df_loss)
# plt.title("Label-wise Loss Distribution (Summary vs Original)")
# plt.ylabel("Loss")
# plt.grid(True, linestyle='--', alpha=0.4)
# plt.tight_layout()
# plt.show()
#
# """시각화 (각 문서별 클러스터링해 히트맵 재배열)"""
# plt.figure(figsize=(12, 6))
# sns.clustermap(df_loss.T, cmap="YlOrRd", figsize=(14, 6), col_cluster=True, row_cluster=False)
# plt.title("Per-Document Label-wise Distribution Loss (Summary vs Original)")
# plt.xlabel("Document Index")
# plt.ylabel("Label")
# plt.tight_layout()
# plt.show()
#
# """시각화 (각 문서별 t-SNE, PCA)"""
# tsne = TSNE(n_components=2, perplexity=30, random_state=42)
# tsne_result = tsne.fit_transform(loss_matrix)
#
# plt.figure(figsize=(10, 6))
# plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=np.mean(loss_matrix, axis=1), cmap="coolwarm", s=10)
# plt.title("Document-level Loss Distribution (t-SNE Projection)")
# plt.colorbar(label="Mean Loss")
# plt.xlabel("t-SNE 1")
# plt.ylabel("t-SNE 2")
# plt.grid(True, linestyle='--', alpha=0.3)
# plt.tight_layout()
# plt.show()
#
# """시각화 (Histogram)"""
# plt.figure(figsize=(14, 6))
#
# metrics = ['precision', 'recall', 'f1']
# bins = 10  # 원하는 구간 수
#
# for i, metric in enumerate(metrics, 1):
#     plt.subplot(1, 3, i)
#     plt.hist(df_doc[metric], bins=bins, edgecolor='black', alpha=0.7)
#     plt.title(f"{metric.capitalize()} Distribution")
#     plt.grid()
#     plt.xlabel("Score")
#     plt.ylabel("Frequency")
#     plt.ylim(0, len(df_doc))  # y축 최대치를 문서 수로 제한
#
# plt.suptitle("Histogram of Precision, Recall, F1 Across Documents")
# plt.tight_layout(rect=(0, 0.03, 1, 0.95))
# plt.show()

if __name__ == "__main__":
    evaluate(
        summary_path = summary_path_4o,
        summary_data = summary_data_4o,
        label_stats = label_stats_4o,
        label_freq_stats = label_freq_stats_4o,
        doc_stats = doc_stats_4o,
        false_positives = false_positives_4o,
        loss_vectors = loss_vectors_4o
    )
    evaluate(
        summary_path=summary_path_35,
        summary_data=summary_data_35,
        label_stats=label_stats_35,
        label_freq_stats=label_freq_stats_35,
        doc_stats=doc_stats_35,
        false_positives=false_positives_35,
        loss_vectors=loss_vectors_35
    )
    df_doc_4o = init_dataframe_docs(doc_stats_4o)
    df_doc_35 = init_dataframe_docs(doc_stats_35)
    # cdf_prec_recall(df_doc_4o, df_doc_35)
    freq_based_entity_similarity(label_freq_stats_4o, label_freq_stats_35)