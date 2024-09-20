from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Load saved model and tokenizer
model = BertForSequenceClassification.from_pretrained(r'Z:\Python_Dataanalytics project\SafeSpeak\Toxic Model')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#new_comment = "that was good point"
# new_comment = "go to hell"
new_comment=input("Comment : ")
encoded_comment = tokenizer(new_comment, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
output = model(**encoded_comment)
probabilities = torch.nn.functional.softmax(output.logits, dim=-1)
predicted_label = torch.argmax(probabilities, dim=-1)
if predicted_label == 0:
    print("Toxic comment.")
else:
    print("Non-Toxic comment")

print("Predicted Label:", predicted_label.item())
