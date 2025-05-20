import pandas as pd
import os

# Proje kök dizinine göre yol oluşturma
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Backend dizininden yukarı çık
DB_DIR = os.path.join(PROJECT_DIR, "database")
DB_PATH = os.path.join(DB_DIR, "users.xlsx")

def initialize_database():
    """
    Kullanıcı veritabanını başlatır.
    Eğer Excel dosyası yoksa veya 'database' klasörü yoksa yeni bir tane oluşturur.
    """
    # Klasör yoksa oluştur
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)

    # Excel dosyası yoksa oluştur
    if not os.path.exists(DB_PATH):
        df = pd.DataFrame(columns=["username", "password"])
        df.to_excel(DB_PATH, index=False)

def authenticate_user(username, password):
    """
    Kullanıcı giriş bilgilerini doğrular ve ad soyad bilgisini döndürür.
    """
    df = pd.read_excel(DB_PATH)

    # Veri temizliği ve tür dönüşümü
    df["username"] = df["username"].apply(lambda x: str(x).strip())
    df["password"] = df["password"].apply(lambda x: str(x).strip())
    df["name"] = df["name"].apply(lambda x: str(x).strip())

    username = str(username).strip()
    password = str(password).strip()

    # Kullanıcı doğrulama
    user = df[(df["username"] == username) & (df["password"] == password)]
    if not user.empty:
        name = user["name"].values[0]
        return True, "Giriş başarılı.", name
    return False, "Kullanıcı adı veya şifre hatalı.", None


