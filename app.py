import streamlit as st

st.set_page_config(layout="wide")

#----PAGE SETUP----
about_page = st.Page(
    page = "pages/about_me.py",
    title = "About Me",
    icon = ":material/account_circle:",
    default = True,
)
#----PAGE PROJECT-----
dashboard_page = st.Page(
    page ='pages/dashboard.py',
    title = "Dashboard",
    icon = ":material/bar_chart:"
)
page_model = st.Page(
    page = 'pages/predict.py',
    title = "Prediksi Produksi Padi",
    icon = ":material/smart_toy:"
)

#----NAVIGATION PAGE SETUP WITH SECTION-----
pg = st.navigation(
    {
        "Info": [about_page],
        "Projects": [dashboard_page, page_model],
    }
)

#----LOGO-------
#st.logo('images/')
st.sidebar.text(f'Created by Fendy Hendriyanto 👨🏼‍💻')
st.sidebar.info("Untuk codingnya, bisa ditemukan di my GitHub:")
st.sidebar.link_button("GitHub Source"," ")
#---- RUN NAVIGATION ------
pg.run()