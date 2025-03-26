import os
import pandas as pd

#path folder ke dataset
dataset_path = r"D:\ROSA\Coolyeah\Pengenalan Pola\tugas segmentasi sel darah\dataset"
images_folder = os.path.join(dataset_path, "images")
masks_folder = os.path.join(dataset_path, "masks_origin")

#mengmbil daftar file dari folder images dan masks
images_file = sorted(os.listdir(images_folder))
masks_file = sorted(os.listdir(masks_folder))

#memastikan jumlah file sama dan sesuai pasangannya
assert len(images_file) == len(masks_file), "Jumlah images dan masks tidak sama"


#Membuat list pasangan path image dan mask
data = []
for img_file, mask_file in zip(images_file, masks_file):
    img_path = os.path.join(images_folder, img_file)
    mask_path = os.path.join(masks_folder, mask_file)
    data.append([img_path, mask_path])
    
#Simpan ke dalam dataframe pandas
df = pd.DataFrame(data, columns=["image", "mask"])

#disimpan sebagai file csv
csv_path = "dataset.csv"
df.to_csv(csv_path, index=False)

print(f"File CSV telah dibuat: {csv_path}")
