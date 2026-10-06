import csv
import ujson
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score

# 파일 경로
original_path = "original_ner.json" # 원문으로부터 CyNER를 통해 추출한 Label
llm_qa_data_path = "gpt_results_250427.json" # LLM을 통해 QA한 main NER Entity

# JSON 로드
with open(original_path, "r") as f:
    original_data = ujson.load(f)
with open(llm_qa_data_path, "r") as f:
    llm_qa_data = ujson.load(f)

# NER 레이블 종류
LABELS = ["Malware", "System", "Indicator", "Vulnerability", "Organization"]

# id와 qa_text만 추출하는 딕셔너리 생성
id_to_qa = {item["id"]: item["qa_text"] for item in llm_qa_data if "id" in item and "qa_text" in item}

processed_qa = {}

for id_, qa_str in id_to_qa.items():
    # 중괄호 제거, 양끝 공백 제거 후 ',' 기준으로 split
    items = qa_str.strip("{} ").split(',')
    # 각 항목에서 양끝 공백 제거
    items = [item.strip() for item in items if item.strip()]
    processed_qa[id_] = items

def parse_ner_string(ner_string, labels):
    """
    예시 입력: "{LockBit, no answer, FakeUpdates, Evil Corp}"
    출력: {label: entity or 'no answer'}
    """
    ner_string = ner_string.strip("{} ")
    elements = [element.strip() for element in ner_string.split(",")]

    # 길이 안 맞으면 그냥 패스 (데이터 불완전할 경우)
    if len(elements) != len(labels):
        return None

    return dict(zip(labels, elements))

import csv

def save_failed_predictions(processed_qa, original_ner_dict, output_path="failed_cases_gpt4omini.csv"):
    with open(output_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label", "prediction", "gold_entities", "match"])

        for id_, pred_entities in processed_qa.items():
            ner_entry = original_ner_dict.get(str(id_))
            if not ner_entry:
                continue

            try:
                for idx, label in enumerate(LABELS):
                    if idx == 2:
                        # 현재 label이 indicator이면 vulnerability로 교체
                        pred = pred_entities[3]
                    elif idx == 3:
                        # 현재 label이 vulnerability면 indicator로 교체
                        pred = pred_entities[2]
                    else:
                        pred = pred_entities[idx]
                    gold_entities = set(map(str.lower, ner_entry.get(label, [])))

                    pred_set = set() if pred.lower() == "no answer" else {pred.lower()}
                    match = bool(pred_set & gold_entities)

                    if not match and gold_entities:
                        writer.writerow([id_, label, pred, list(gold_entities), match])
            except Exception as e:
                print(f"[SAVE-FAILED-PRED] 문서 번호 : {id_}, 오류 예측 QA : {pred_entities}")
                continue


# 예: processed_qa = {0: ['LockBit', 'no answer', ...], ...}
# 예: original_ner = ["0": {"System": [...], "Malware": [...], ...}, "1": {...}]
def evaluate_qa_vs_ner(processed_qa, original_ner_dict):
    y_true = defaultdict(list)
    y_pred = defaultdict(list)

    for id_, pred_entities in processed_qa.items():
        ner_entry = original_ner_dict.get(str(id_))
        if not ner_entry:
            continue

        try:
            for idx, label in enumerate(LABELS):
                if idx == 2:
                    # 현재 label이 indicator이면 vulnerability로 교체
                    pred = pred_entities[3]
                elif idx == 3:
                    # 현재 label이 vulnerability면 indicator로 교체
                    pred = pred_entities[2]
                else:
                    pred = pred_entities[idx]
                gold_entities = set(map(str.lower, ner_entry.get(label, [])))

                # if pred.lower() in
                if pred.lower() != "no answer":
                    pred_set = {pred.lower()}
                else:
                    pred_set = set()


                match = bool(pred_set & gold_entities)
                # TODO: true, pred 수집 방식을 바꿔봐야 할듯
                y_true[label].append(1 if match else 0) # 예측해야 할 엔티티가 골드 엔티티에 포함된 경우
                y_pred[label].append(1 if pred_set else 0) # 예측해야 할 엔티티가 "no answer"이 아닌 의미있는 엔티티인 경우
                if y_true[label][-1] == 1 and y_pred[label][-1] == 0:
                    print(f"문서 번호 : {id_}, pred_entitiy: {pred}")

        except Exception as e:
            print(f"문서 번호 : {id_}, 오류 예측 QA : {pred_entities}")
            continue

    # 결과 출력
    print("Evaluation Results:")
    print(f"Original Articles : {len(original_data)}")
    print(f"LLM QA Articles : {len(llm_qa_data)}")
    for label in LABELS:
        p = precision_score(y_true[label], y_pred[label], zero_division=0)
        r = recall_score(y_true[label], y_pred[label], zero_division=0)
        f1 = f1_score(y_true[label], y_pred[label], zero_division=0)
        print(f"{label:15s} | Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f} | 총 개수: {len(y_true[label])}")

    all_p = [precision_score(y_true[label], y_pred[label], zero_division=0) for label in LABELS]
    all_r = [recall_score(y_true[label], y_pred[label], zero_division=0) for label in LABELS]
    all_f1 = [f1_score(y_true[label], y_pred[label], zero_division=0) for label in LABELS]
    print(f"\n{'문서 총 평균':15s} | Precision: {sum(all_p)/len(LABELS):.2f} | Recall: {sum(all_r)/len(LABELS):.2f} | F1: {sum(all_f1)/len(LABELS):.2f}")


if __name__ == '__main__':
    evaluate_qa_vs_ner(processed_qa, original_data)
    save_failed_predictions(processed_qa, original_data)