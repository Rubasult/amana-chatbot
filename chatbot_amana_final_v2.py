
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import json

# تحميل البيانات من ملف JSON
with open("faq_data_fixed.json", "r", encoding="utf-8") as file:
    faq_data_fixed = json.load(file)

# تحميل نموذج التحويل
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# تجهيز قائمة الأسئلة لجميع الخدمات
all_questions = []
question_to_service = {}

for service in faq_data_fixed:
    question = service["question"]
    all_questions.append(question)
    question_to_service[question] = service
# حساب المتجهات لجميع الأسئلة
question_embeddings = model.encode(all_questions, convert_to_tensor=True)

# واجهة المستخدم
st.image("hail_logo.png", width=120)
st.markdown("<h1 style='text-align: center; color: green;'>شات بوت خدمات أمانة منطقة حائل</h1>", unsafe_allow_html=True)
st.markdown("اسأل عن أي خدمة من خدمات الأمانة: إيجاد، عونك، غرس، المستكشف الجغرافي 🌱")

user_question = st.text_input("✍️ اكتب سؤالك هنا:")

if user_question.strip() == "":
    st.info("👋 مرحبًا بك! يمكنك كتابة سؤالك حول خدمات أمانة منطقة حائل، مثل: ما هي خدمة عونك؟ أو كيف أصل إلى خدمة غراس في توكلنا؟")

elif user_question.strip() != "":
    user_embedding = model.encode(user_question, convert_to_tensor=True)
    cos_scores = util.cos_sim(user_embedding, question_embeddings)[0]
    top_result = torch.argmax(cos_scores).item()
    matched_question = all_questions[top_result]
    matched_service = question_to_service[matched_question]
    score = cos_scores[top_result].item()

    if score > 0.65:
        st.markdown(f"### ✅ {matched_service['service']}")
        st.success(matched_service["answer"])
        st.info("🧭 خطوات الوصول للخدمة عبر تطبيق توكلنا:\n\n" + matched_service["steps"])
    else:
        st.warning("❗لم يتم العثور على خدمة مطابقة بدقة. حاول إعادة صياغة سؤالك.")
