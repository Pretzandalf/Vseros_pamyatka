# Vseros_pamyatka
For me


model.config.num_labels = 1  # Для предсказания одного значения (вероятности)
model.lm_head = nn.Linear(model.config.n_embd, 1)  # Изменяем последний слой на линейный
