import streamlit as st

col1, col2 = st.columns(2, gap='small', vertical_alignment='center')
with col1:
    st.image("images/Fendy.png", width=250)
with col2:
    st.title("Fendy Hendriyanto", anchor=False)
    st.write(
        "AI Researcher & Instructor | ML Engineer"
    )
    st.write(
        "Assisting and mentoring students to help analyze and supporting data driven with creativity and decision making."
    )

#--- EXPERIENCE & QUALIFICATIONS ------
st.write("\n")
st.subheader("Experience and Qualifications", anchor=False)
st.write(
    """
    - 3 years experience coaching and mentoring about Artificial Intelligence and Data Science
    - Strong hands-on experience and knowledge in Python programming and Data Science
    - Proficient in using various libraries and tools such as TensorFlow, Keras, Scikit-learn, OpenCV, Pandas
    - Good understanding and analyzing of statistical principles and their perspective applications
    - Excellent team player and initiative on tasks

    """
)

# ---- SKILLS ----
st.write("\n")
st.subheader("Hard Skills", anchor=False)
st.write(
    """
    - Programming Languages : Python, R, SQL, JavaScript
    - Data Analysis and Visualization : Pandas, Matplotlib, Seaborn, Tableau, Spreadsheet, Excel
    - Modelling : Tensorflow, Keras, PyCaret, PyTorch, XGBoost, CometML, Scikit-learn
    - Databases : MySQL, PostgreSQL, SQLite
    - Deployment : Streamlit, Flask, Gradio, Huggingface, FastAPI
    - Version Control: Git, GitHub
    - CV and NLP Frameworks : OpenCV, NLTK, Scikit-image, Pillow

    """
)