import pkg_resources

# Daftar paket yang digunakan
packages = ["librosa", "matplotlib", "numpy", "IPython"]

# Simpan versi yang terinstal
with open("requirements.txt", "w") as f:
    for pkg in packages:
        try:
            version = pkg_resources.get_distribution(pkg).version
            f.write(f"{pkg}=={version}\n")
        except pkg_resources.DistributionNotFound:
            f.write(f"{pkg} (not installed)\n")

print("✅ File requirements.txt berhasil dibuat hanya dengan paket yang relevan.")
