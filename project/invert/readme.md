### 图像内插


方法一：近邻内插法
以新网格如 700x700网格覆盖元图像，新网格中的像素取图像中最接近的像素

缺点：边缘失真

方法二： 线性内插法
双线性内插法
$$ f(x,y) = ax + by + cxy + d $$

三次内插法
$$ f(x,y) = \sum^{3}_{i =0}    \sum^{3}_{j=0}   ( a_{ij}* x^i * y^j)   $$

