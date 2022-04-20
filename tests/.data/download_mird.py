import urllib.request

if __name__ == "__main__":
    filename = "Impulse_response_Acoustic_Lab_Bar-Ilan_University__"
    "Reverberation_0.160s__8-8-8-8-8-8-8.zip"
    url = "https://www.iks.rwth-aachen.de/fileadmin/user_upload/downloads/"
    "forschung/tools-downloads/{}".format(filename)
    save_name = "./data/{}".format(filename)

    urllib.request.urlretrieve(url, save_name)
