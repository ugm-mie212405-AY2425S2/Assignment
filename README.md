# EL Class Pattern Recognition Assignment Repo

## Petunjuk Bagi Mahasiswa

Untuk mengerjakan tugas, ikuti langkah-langkah berikut:

1. **Fork Repository Ini**  
   - Klik tombol **Fork** di sudut kanan atas halaman repository ini untuk membuat salinan repository ke akun GitHub Anda.

2. **Clone Fork Repository Anda ke Local**  
   - Buka terminal dan jalankan perintah berikut (ganti `<username>` dengan username GitHub Anda):
     ```bash
     git clone https://github.com/<username>/Assignment.git
     ```
   
3. **Buat Branch di Fork Anda**  
   - Masuk ke direktori repository yang telah di-clone:
     ```bash
     cd Assignment
     ```
   - Buat branch baru dengan nama `Assignment-n`, di mana `n` adalah nomor tugas. Contoh: untuk Tugas 1, gunakan:
     ```bash
     git checkout -b Assignment-1
     ```

4. **Kerjakan Tugas Anda**  
   - Tambahkan file-file hasil pengerjaan tugas Anda, baik file **.ipynb** maupun **.py** ke dalam branch yang telah dibuat.

5. **Push File ke Fork Repository Anda**  
   - Setelah selesai mengerjakan, jalankan perintah berikut untuk mengirim perubahan ke GitHub:
     ```bash
     git add .
     git commit -m "Tugas 1 - Penugasan EL Class Pattern Recognition"
     git push origin Assignment-1
     ```

6. **Lakukan Pull Request ke Repository Utama (Upstream)**  
   - Buka repository fork Anda di GitHub.
   - Klik tombol **Compare & pull request** pada branch `Assignment-1`.
   - Buat pull request ke repository utama.
   - Tunggu hingga seluruh pemeriksaan (checks) selesai sebelum pull request Anda di-merge.

7. **Selesai**  
   - Setelah pull request diterima dan semua pemeriksaan selesai, tugas Anda dianggap selesai. Langkah anda selesai sampai di Pull Request!

Selamat mengerjakan!
