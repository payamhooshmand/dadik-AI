import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import arabic_reshaper
from bidi.algorithm import get_display

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load tokenizer and model
model_name = "C:/Users/Dadik/desktop/fine_tuned_model13"  # Update this if the model path changes
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)  # Use the fast tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

# Convert text for proper Farsi display
def convert(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

# Initialize the pipeline with the fine-tuned model
pipe = pipeline(
    "text2text-generation",
    model=model, 
    tokenizer=tokenizer, 
    device=device,  # Use the selected device
    max_length=300,
    batch_size=1
)

# The text we want to generate questions from, the user's input
user_input = "ماده(11) بانک مرکزی جمهوری اسلامی ایران موظف است با همکاری سازمان ظرف مدت یک سال پس از ابلاغ این قانون نسبت به ساماندهی دستگاههای کارتخوان بانکی و یا در گاههای پرداخت الکترونیکی اقدام نموده و با ایجاد مناظر بین آن ها با مجوز فعالیت و شماره اقتصادی بنگاههای اقتصادی به هر یک از پایانه های فروش شناسه یکتا اختصاص دهد پس از تخصیص شناسه مذکور کلیه تراکنشهای انجام شده از طریق حساب های بانکی متصل به دستگاههای کارتخوان بانکی و نیز در گاههای پرداخت الکترونیکی به عنوان تراکنشهای بانکی مرتبط با فعالیت شغلی صاحب حساب بانکی محسوب شده و بانک مرکزی جمهوری اسلامی ایران موظف است در چهارچوب قانون و مقررات مربوطه اطلاعات این تراکنشهای بانکی شامل مانده اول دوره وجود و اریزی و جوه برداشت شده و مانده آخر دوره در حساب بانکی را به منظور تکمیل پایگاه اطلاعات هویتی عملکردی و دارایی مودیان موضوع ماده ( 169 مکرر ) قانون مالیات های مستقیم مصوب سوم اسفند هزار و سیصد و شصت و شش با اصلاحات و الحاقات بعدی به صورت برخط در اختیار سازمان قرار دهد تبصره پس از انقضای مواد ( گذشت زمان ) مذکور در این ماده اتصال دستگاههای کارتخوان بانکی  و یا در گاههای پرداخت الکترونیکی که تعلق آن ها به مودی معین توسط سازمان امور مالیاتی تایید نشده باشد به شبکه پرداخت بانکی کشور ممنوع است بانک مرکزی و حسب مورد کلیه بانکها و ارایه دهندگان خدمات پرداخت موظف هستند مشخصات بهره برداران کلیه دستگاههای کارتخوان بانکی  و پایانه های پرداخت الکترونیکی را به سازمان اعلام کنند در صورت تخلف از حکم این ماده مرتکبان به مجازات درجه شش قانون مجازات اسلامی به غیر از حبس محکوم می شوند"

num_questions = 10  # Number of questions to generate

# Function to generate rephrased questions (using the same model)
def generate_rephrased_questions(input_text, num_rephrases=10):
    rephrased_questions = []
    for _ in range(num_rephrases):
        try:
            # Generate a question based on the input text
            generated_question = pipe([input_text],
                                      temperature=0.7,  # Slight randomness to encourage variety
                                      do_sample=True,
                                      repetition_penalty=2.0,
                                      max_length=150,  # Adjust based on your desired question length
                                      top_k=50,
                                      top_p=0.9,
                                )
            question = generated_question[0]['generated_text']
            
            # Split the question into separate ones based on '?'
            split_questions = question.split(',')
            for q in split_questions:
                if q.strip():  # Avoid empty strings
                    rephrased_questions.append(q.strip() + "")
        except Exception as e:
            print(f"Error generating question: {e}")
            rephrased_questions.append("Error generating question.")
    return rephrased_questions

# Generate rephrased questions
generated_questions = generate_rephrased_questions(user_input, num_rephrases=num_questions)

# Print the input text and the generated questions
print("Input Text:", convert(user_input))
print("Generated Questions:")
print()
for i, question in enumerate(generated_questions, 1):
    print(f"{i}. {convert(question)}")
    print()
