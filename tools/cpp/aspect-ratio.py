# %%
import math

width = [i for i in range(192, 416+1, 32)]
valid_height = [w / 4.0 * 3.0 for w in width]
height = [math.ceil(h / 32) * 32 for h in valid_height]

# %%
for h, w, vh in zip(height, width, valid_height):
    print(h, w, vh, h-vh, (h-vh) * w)

# %%
