# Regressions


1. Regressiya nima?
Regressiya — bu statistik usul, yordamida biror bog‘liq o‘zgaruvchini (masalan, narx, daromad yoki harorat) mustaqil o‘zgaruvchilar (masalan, vaqt, maydon, tezlik) bilan bog‘lash mumkin. Bu usul odatda bashorat qilish yoki tahlil qilish uchun qo‘llanadi.
Oddiy ta'rif:
Regressiya — ma'lumotlar to‘plamida mavjud bo‘lgan qonuniyat va munosabatni o‘rganish, so‘ngra undan yangi kiruvchi ma'lumotlar uchun natijalarni oldindan aytib berishdir.

2. Nima uchun regressiya kerak?
Regressiya asosan quyidagi sabablarga ko‘ra ishlatiladi:
a) Bashorat qilish
    • Uy yoki avtomobil narxini bashorat qilish. 
    • O‘simlik hosildorligini tuproq xususiyatlariga qarab hisoblash. 
    • Yoqilg‘i narxi kelajakda qancha bo‘lishini aniqlash. 
b) Munosabatni tushunish
    • Xususiyatlar o‘rtasidagi bog‘liqlikni o‘rganish. Masalan: 
        ◦ Uy narxi uy maydoni va joylashuvga qanday bog‘liq? 
        ◦ Harorat va muzqaymoq sotilishi o‘rtasidagi munosabat. 
c) Tushunchani soddalashtirish
    • Ko‘p o‘zgaruvchili vaziyatlarda, asosiy ta'sir ko‘rsatuvchi omillarni aniqlash. 

3. Regressiyaning turlari
Regressiya modellari turli-tuman bo‘lib, ular ma'lumotlarning murakkabligi va xarakteriga qarab tanlanadi.
a) Chiziqli regressiya (Linear Regression)
    • Bu eng oddiy va eng ko‘p ishlatiladigan regressiya turi. 
    • Maqsadi: bog‘liq o‘zgaruvchi (y) bilan mustaqil o‘zgaruvchi (x) o‘rtasida to‘g‘ri chiziqni topish. 
        ◦ Tenglama: y=m⋅x+by 
          Bu yerda: 
            ▪ y — bashorat qilinadigan qiymat. 
            ▪ m — chiziqning qiyaligi (slop). 
            ▪ b — kesish nuqtasi (intercept). 
Misol:
Uy maydoni (x) va uning narxi (y) o‘rtasidagi to‘g‘ri chiziqli munosabat.

b) Polinomial regressiya (Polynomial Regression)
    • Ma'lumotlar chiziqli emas (egri chiziqli) bo‘lganda qo‘llaniladi. 
    • Polinom darajalarini qo‘shish orqali model yanada moslashuvchan bo‘ladi. 
        ◦ Tenglama: y=a0+a1⋅x+a2⋅x2+...+an⋅xn
           Bu yerda nn — polinom darajasi. 
Misol:
Harorat va shamol tezligi o‘rtasida murakkab bog‘liqlikni o‘rganish.

c) Ko‘p o‘zgaruvchili regressiya (Multiple Linear Regression)
    • Bir nechta mustaqil o‘zgaruvchilar ishlatiladi. 
        ◦ Tenglama: y=b0+b1⋅x1+b2⋅x2+...+bn⋅xny 
           Bu yerda: 
            ▪ x1,x2,...,xnx_1, x_2, ..., x_n — mustaqil o‘zgaruvchilar. 
            ▪ y — bashorat qilinadigan qiymat. 
Misol:
Uy narxi (y) maydon (x1), xonalar soni (x2) va joylashuv (x3)ga bog‘liq.

d) Logistik regressiya (Logistic Regression)
    • Asosan tasniflash (classification) uchun ishlatiladi. 
    • Y o‘zgaruvchisi diskret qiymatlarga ega, masalan: 0 yoki 1 (ha yoki yo‘q). 
        ◦ Tenglama: P(y=1)=11+e−(b0+b1⋅x1+b2⋅x2+...+bn⋅xn)
          
Misol:
Kredit olish imkoniyati (1 yoki 0) mijozning daromadiga va kredit tarixiga bog‘liq.

Tasodifiy dataset yaratish.
![image](https://github.com/user-attachments/assets/7b693390-cbd9-421e-8dbd-6ca42ec4b049)




Bir o’zgaruvchili regressiya modeli
![image](https://github.com/user-attachments/assets/ad7fbee1-814a-45aa-9d13-19af52f0304c)


Har-bir polinom darajalari uchun regressiya grafigi.
![Figure_1](https://github.com/user-attachments/assets/17c1b9bd-f9f9-48ac-8922-0f3c838f989b)





 Modellarni testlash.  
![image](https://github.com/user-attachments/assets/a42cd94a-efdf-4b6f-8a21-ac08b29ec93e)
bir o’zgaruvchili va ko’p o’zgaruvchili regressiya test natijalari.

Test natijalarini taqqoslash (MSE va R2)
Bir o’zgaruvchili model natijalari: Polinom darajasi oshgani sari MSE kamayishi, lekin haddan tashqari moslashuv (overfitting) yuzaga kelish havfi oshiadi.
Ko’p o’zgaruvchili model natijalari: Malumotlarni yaxshiroq tushuntiradi chunki ko’p omillar inobatga olinadi. Lekin yuqori polinom darajalarida overfitting ehtimoli oshadi.

 Xulosa
Regressiya modellari:
    1. Haqiqiy hayotdagi ma'lumotlar o‘rtasidagi bog‘liqlikni tushunishga yordam beradi. 
    2. Har xil vaziyatlarda bashorat qilish imkoniyatini beradi. 
    3. Polinomial regressiya kabi usullar murakkab bog’liqliklarni yaxshiroq ifodalashi mumkin. 


