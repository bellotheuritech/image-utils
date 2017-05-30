from imageutils import imageutils

img_name = '00ed9bf0d1306ec4c57713f675a1798e'

img = imageutils.load_img(img_name)

assert(img is not None)

md5 = imageutils.get_md5_from_imagepath(img_name)

assert(img_name == md5)

assert(img.size == imageutils.load_img_from_md5(md5).size)

assert(imageutils.load_img('fake_img_path') is None)

print('Everything OK!')
