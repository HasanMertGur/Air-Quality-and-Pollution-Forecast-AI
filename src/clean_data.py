import pandas as pd
import numpy as np

def clean_air_quality_data(input_file, output_file):
    print(f"{input_file} okunuyor...")
    df = pd.read_csv(input_file)
    
    print(f"Orijinal veri boyutu: {df.shape}")

    # 1. Eksik Değerler (Missing Values)
    # Hedef değişkenimiz (örn: pm2.5) boşsa o satırı atıyoruz
    if 'pm2.5' in df.columns:
        df.dropna(subset=['pm2.5'], inplace=True)
    # Diğer sütunlardaki boşlukları bir önceki geçerli değerle dolduruyoruz
    df.ffill(inplace=True)

    # 2. Kopyalanmış Veriler (Duplicates)
    df.drop_duplicates(inplace=True)

    # 3. Aykırı Değerler (Outliers) - IQR (Çeyrekler Arası Aralık) Yöntemi
    # Sadece sayısal sütunlarda işlem yapıyoruz
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Aykırı değerlerin dışındaki verileri filtrele
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    print(f"Temizlenmiş veri boyutu: {df.shape}")
    
    # Temizlenmiş veriyi kaydet
    df.to_csv(output_file, index=False)
    print(f"Temizlenmiş veri {output_file} olarak başarıyla kaydedildi!")

if __name__ == "__main__":
    # Arkadaşının kullandığı ham veri dosyasının adını 'raw_data.csv' yerine yazabilirsin
    clean_air_quality_data('raw_data.csv', 'cleaned_data.csv')