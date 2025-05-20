import tkinter as tk
from tkinter import messagebox, Label, StringVar
from PIL import Image, ImageTk
from gesture_recognition.backend import user_management, analyzer  # Analyzer ve user_management modülleri
import datetime
import time

# Veritabanını başlat
user_management.initialize_database()

def get_current_time():
    """
    Mevcut saati döndürür.
    """
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def update_time_label(time_label_var, time_label):
    """
    Saat ve tarih etiketlerini günceller.
    """
    time_label_var.set(get_current_time())
    time_label.after(1000, lambda: update_time_label(time_label_var, time_label))

def start_video_processor(main_screen_log):
    """
    Video işleme ekranını başlatır.
    """
    # Yeni pencere oluştur
    video_window = tk.Toplevel()
    video_window.title("El Hareketi Algılama")
    video_window.geometry("800x600")

    # Tkinter'de görüntüyü göstermek için bir Label ekle
    video_label = Label(video_window)
    video_label.pack()

    # Algılanan sonuçları göstermek için bir Label
    status_label = Label(video_window, text="Durum: Bekleniyor", font=("Helvetica", 16))
    status_label.pack()

    # Durum değişkenleri
    detected_four = False  # 4 parmak algılandığını takip eder
    four_detected_time = None  # 4 parmağın algılandığı zamanı tutar
    time_threshold = 2  # 4 parmak algılandıktan sonra izin verilen süre (saniye)

    def update_frame():
        nonlocal detected_four, four_detected_time
        # Backend'den görüntü ve parmak sayısını al
        frame, finger_count = analyzer.process_frame()

        if frame is None:
            return

        current_time = time.time()  # Mevcut zamanı al

        # Eğer 4 parmak algılandıysa
        if finger_count == 4 and not detected_four:
            detected_four = True
            four_detected_time = current_time  # 4 parmağın algılandığı zamanı kaydet
            status_label.config(text="Durum: 4 Parmak Algılandı!", fg="green")
            main_screen_log.insert(tk.END, f"[{get_current_time()}] Durum: 4 Parmak Algılandı!\n")
            main_screen_log.see(tk.END)

        # Eğer el kapanırsa (0 parmak) ve 4 parmak kısa süre önce algılandıysa
        elif finger_count == 0 and detected_four:
            if current_time - four_detected_time <= time_threshold:  # Zaman kontrolü
                print("UYARI: 4 parmak yaptıktan hemen sonra yumruk algılandı!")
                status_label.config(text="Durum: UYARI! Yumruk Algılandı!", fg="red")
                main_screen_log.insert(tk.END, f"[{get_current_time()}] UYARI: 4 parmak yaptıktan hemen sonra yumruk algılandı!\n")
                main_screen_log.see(tk.END)
            else:
                print("Yumruk çok geç yapıldı veya başka bir işaret algılandı.")
                status_label.config(text="Durum: Geçersiz İşlem", fg="orange")
                main_screen_log.insert(tk.END, f"[{get_current_time()}] Yumruk çok geç yapıldı veya başka bir işaret algılandı.\n")
                main_screen_log.see(tk.END)
            detected_four = False  # Durumu sıfırla

        # Eğer 4 parmak yapıldıktan sonra başka bir işaret yapılırsa
        elif finger_count not in [4, 0] and detected_four:
            print("4 parmak yapıldıktan sonra başka bir işaret yapıldı.")
            status_label.config(text="Durum: Geçersiz İşlem", fg="orange")
            main_screen_log.insert(tk.END, f"[{get_current_time()}] 4 parmak yapıldıktan sonra başka bir işaret yapıldı.\n")
            main_screen_log.see(tk.END)
            detected_four = False  # Durumu sıfırla

        # OpenCV görüntüsünü Tkinter'e aktar
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        # Döngüyü tekrar çalıştır
        video_label.after(10, update_frame)

    # Kamerayı güncellemeyi başlat
    update_frame()

    # Kapatma
    video_window.protocol("WM_DELETE_WINDOW", lambda: [analyzer.release_resources(), video_window.destroy()])

def open_main_screen(user_name):
    """
    Ana ekranı açar.
    """
    login_frame.pack_forget()

    main_screen = tk.Frame(root, bg="#f0f8ff")
    main_screen.pack(fill="both", expand=True)

    # Logo Ekle
    logo_img_main = Image.open("../assets/eskisehir-teknik-universitesi_logo.png")  # Logo dosya yolunu kontrol edin
    logo_img_main = logo_img_main.resize((150, 150), Image.LANCZOS)
    logo_photo_main = ImageTk.PhotoImage(logo_img_main)
    logo_label_main = tk.Label(main_screen, image=logo_photo_main, bg="#f0f8ff")
    logo_label_main.image = logo_photo_main
    logo_label_main.pack(pady=10)

    # Başlık
    title_label = Label(main_screen, text=f"Welcome to the Main Application, {user_name}", font=("Helvetica", 16, "bold"), bg="#f0f8ff")
    title_label.pack(pady=10)

    # Tarih ve Saat
    time_label_var = StringVar()
    time_label_var.set(get_current_time())
    time_label = Label(main_screen, textvariable=time_label_var, font=("Helvetica", 12), bg="#f0f8ff")
    time_label.pack(pady=10)
    update_time_label(time_label_var, time_label)

    # Log Alanı
    main_screen_log = tk.Text(main_screen, height=10, width=80, state=tk.NORMAL)
    main_screen_log.pack(pady=10)

    # Uygulamayı Başlat Butonu
    start_button = tk.Button(main_screen, text="Uygulamayı Başlat", font=("Helvetica", 12), bg="#4CAF50", fg="black",
                              command=lambda: start_video_processor(main_screen_log))
    start_button.pack(pady=10)

    # Çıkış Butonu
    logout_button = tk.Button(main_screen, text="Logout", font=("Helvetica", 12), bg="#f0f8ff", fg="black",
                               command=lambda: [main_screen.pack_forget(), login_frame.pack(fill="both", expand=True)])
    logout_button.pack(pady=10)

def open_main_application(event=None):
    """
    Kullanıcı giriş bilgilerini doğrular ve ana ekranı açar.
    """
    username = username_entry.get()
    password = password_entry.get()

    # Kullanıcı giriş doğrulaması
    success, message, user_name = user_management.authenticate_user(username, password)
    if success:
        messagebox.showinfo("Login Successful", f"Welcome, {user_name}!")
        open_main_screen(user_name)  # Ana ekranı aç
    else:
        messagebox.showerror("Login Failed", message)

def toggle_password():
    """
    Şifreyi gizle/göster.
    """
    if password_entry.cget('show') == "*":
        password_entry.config(show="")
        toggle_button.config(text="Hide Password")
    else:
        password_entry.config(show="*")
        toggle_button.config(text="Show Password")

# Tkinter GUI Ayarları
root = tk.Tk()
root.title("University Portal")
root.geometry("800x600")

# Giriş Sayfası
login_frame = tk.Frame(root, bg="#f0f8ff")
login_frame.pack(fill="both", expand=True)

# Logo Ekle
logo_img = Image.open("../assets/eskisehir-teknik-universitesi_logo.png")  # Logo dosya yolunu kontrol edin
logo_img = logo_img.resize((300, 300), Image.LANCZOS)
logo_photo = ImageTk.PhotoImage(logo_img)
logo_label = tk.Label(login_frame, image=logo_photo, bg="#f0f8ff")
logo_label.pack(pady=10)

login_label = tk.Label(login_frame, text="Welcome to the University Portal", font=("Helvetica", 16, "bold"), bg="#f0f8ff")
login_label.pack(pady=10)

username_label = tk.Label(login_frame, text="Username:", font=("Helvetica", 12), bg="#f0f8ff")
username_label.pack(pady=5)
username_entry = tk.Entry(login_frame, width=30)
username_entry.pack(pady=5)
username_entry.focus()  # İmleç otomatik olarak buraya odaklansın

password_label = tk.Label(login_frame, text="Password:", font=("Helvetica", 12), bg="#f0f8ff")
password_label.pack(pady=5)
password_entry = tk.Entry(login_frame, show="*", width=30)
password_entry.pack(pady=5)

toggle_button = tk.Button(login_frame, text="Show Password", font=("Helvetica", 10), bg="#f0f8ff", fg="black", command=toggle_password)
toggle_button.pack(pady=5)

# Login butonu
login_button = tk.Button(login_frame, text="Login", font=("Helvetica", 12, "bold"), bg="#4CAF50", fg="black", width=15, command=open_main_application)
login_button.pack(pady=10)
root.bind('<Return>', open_main_application)  # Enter ile login yap

# Uygulama başlat
root.mainloop()
