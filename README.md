# Vseros_pamyatka
For me


model.config.num_labels = 1  # Для предсказания одного значения (вероятности)
model.lm_head = nn.Linear(model.config.n_embd, 1)  # Изменяем последний слой на линейный



from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def predict_class(text, model_name="DeepPavlov/rubert-base-cased", num_labels=2):
    # Загружаем модель и токенизатор
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Переключаем модель в режим оценки
    model.eval()
    
    # Токенизация текста
    inputs = tokenizer(text, return_tensors="pt")
    
    # Предсказание
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = logits.argmax().item()
    
    return predicted_class

# Пример использования функции
text_example = "Пример текста для классификации"
predicted_class = predict_class(text_example)
print("Предсказанный класс:", predicted_class)
