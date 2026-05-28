import pandas as pd
import numpy as np

def clean_air_quality_data(input_file, output_file):
    print(f"{input_file} okunuyor...")
    df = pd.read_csv(input_file)
    
    print(f"Orijinal veri boyutu: {df.shape}")

    # 1. Eksik Değerler (Missing Values)
    if 'pm2_5' in df.columns:
        df.dropna(subset=['pm2_5'], inplace=True)
    df.ffill(inplace=True)

    # 2. Kopyalanmış Veriler (Duplicates)
    df.drop_duplicates(inplace=True)

    # 3. Aykırı Değerler (Outliers) - SADECE HEDEF DEĞİŞKEN (pm2_5) İÇİN!
    # Bütün sütunları taramak yerine sadece önemli sütundaki uçuk hataları siliyoruz.
    if 'pm2_5' in df.columns:
        Q1 = df['pm2_5'].quantile(0.25)
        Q3 = df['pm2_5'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Sınırlar içindeki temiz verileri tut
        df = df[(df['pm2_5'] >= lower_bound) & (df['pm2_5'] <= upper_bound)]

    print(f"Temizlenmiş veri boyutu: {df.shape}")
    
    # Temizlenmiş veriyi kaydet
    df.to_csv(output_file, index=False)
    print(f"Temizlenmiş veri {output_file} olarak başarıyla kaydedildi!")

if __name__ == "__main__":
    # Arkadaşının kullandığı veri seti dosya yolu
    clean_air_quality_data('raw_data.csv', 'cleaned_data.csv')