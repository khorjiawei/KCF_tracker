import utils
import numpy as np
import cv2


class KCF:

    def __init__(self):

        # Configuration parameters
        self.learn_rate_ = 0.075   # Model learn rate
        self.sigma_ = 0.2          # Feature bandwidth
        # Spatial bandwidth (of the gaussian peak for target location,
        self.s_ = -1
        # this will be updated to be = sqrt(h*w)/10 once size of template is known)
        self.lambda_ = 0.0001      # Regularisation

        # Model
        # (Fourier transform of) Learned weights in kernel space (higher order)
        self.F_alpha_ = np.ndarray([])
        # (Fourier transform of) Learned feature appearance
        self.F_feature_ = np.ndarray([])

        # Things to save
        # Desired gaussian response peak
        self.target_response_ = np.ndarray([])

        # Track box details
        # Bounding box (top left x, top left y, width, height)
        self.image_patch = np.ndarray([])
        self.bbox_ = Bbox()
        self.patch_height = -1
        self.patch_width = -1
        self.window_height = -1
        self.window_width = -1
        self.enlarge_window_factor = 0.5

        # Booleans
        self.is_empty = True
        self.first_training = True

    def init(self, frame: np.ndarray, bbox: tuple) -> None:
        x, y, w, h = bbox
        x = int(x)
        y = int(y)
        w = int(w)
        w = w-(w % 2)+1  # Ensure w is odd
        h = int(h)
        h = h-(h % 2)+1  # Ensure h is odd
        self.bbox_ = Bbox(x, y, w, h)

        # Update track box details
        self.patch_width = self.bbox_.width
        self.patch_height = self.bbox_.height
        self.window_width = int(
            self.patch_width*(1.0+self.enlarge_window_factor))
        self.window_width = self.window_width - \
            (self.window_width % 2)+1  # Ensure width is odd
        self.window_height = int(
            self.patch_height*(1.0+self.enlarge_window_factor))
        self.window_height = self.window_height - \
            (self.window_height % 2)+1  # Ensure height is odd
        # Update feature bandwidth after patch width and height are known
        self.s_ = np.sqrt(self.patch_height*self.patch_width)/10

        # Allow self.update() to be called from now on
        self.is_empty = False

        # Ready for first training
        self.first_training = True

        # Get image patch
        temp_patch = utils.get_sub_window(
            frame, self.bbox_.center(), (self.window_width, self.window_height))
        self.image_patch = self.__preprocess__(temp_patch)

        # Train
        self.target_response_ = utils.get_gaussian_response(
            self.window_width, self.window_height, self.s_)
        self.__train__(self.image_patch, self.target_response_)

    def update(self, new_image: np.ndarray) -> None:
        if (self.is_empty):
            raise Exception("Tracker not initialised with target yet!")

        # Get new subwindow
        new_patch = utils.get_sub_window(
            new_image, self.bbox_.center(), (self.window_width, self.window_height))

        new_patch = self.__preprocess__(new_patch)
        # Detect
        self.__detect__(np.fft.ifft2(self.F_feature_),
                        self.F_alpha_, new_patch)
        # Obtain new patch for newly estimated position
        new_patch = utils.get_sub_window(
            new_image, self.bbox_.center(), (self.window_width, self.window_height))
        new_patch = self.__preprocess__(new_patch)
        # Train on new patch
        self.__train__(new_patch, self.target_response_)
        self.image_patch = np.copy(new_patch)

    def __preprocess__(self, patch: np.ndarray) -> None:
        # Normalise image
        patch = patch/255.0
        patch = patch-np.mean(patch)

        # Multiply with windowing function
        patch = np.multiply(patch, utils.get_hann_window(
            patch.shape[1], patch.shape[0]))
        return patch

    def __train__(self, x: np.ndarray, y: np.ndarray) -> None:

        if (x.shape != y.shape):
            raise Exception(
                f"x (shape={x.shape}) and y (shape={y.shape} have different shapes")

        # Generate kernel (Equation 16)
        k = self.__gen_kernel__(x, x)

        # Solve Kernelised Regularised Least Squares
        # (Equation 7, actually not quite, inverse fourier is not performed here as in the detection step we still need to calculate F(alpha)
        F_alpha_new = np.fft.fft2(y)/(np.fft.fft2(k)+self.lambda_)
        F_feature_new = np.fft.fft2(x)
        if (self.first_training):
            self.F_alpha_ = F_alpha_new
            self.F_feature_ = F_feature_new

            self.first_training = False
        else:
            self.F_alpha_ = (1-self.learn_rate_)*self.F_alpha_ + \
                self.learn_rate_*F_alpha_new
            self.F_feature_ = (1-self.learn_rate_)*self.F_feature_ + \
                self.learn_rate_*F_feature_new

    def __detect__(self, x: np.ndarray, F_alpha: np.ndarray, z: np.ndarray) -> None:

        # Generate kernel
        k_bar = self.__gen_kernel__(z, x)

        response = np.real(np.fft.ifft2(
            np.multiply(np.fft.fft2(k_bar), F_alpha)))

        # Find location of max response (The indices of max response corresponds to dx and dy of new target location)
        dy, dx = np.unravel_index(response.argmax(), response.shape)
        if dy > (z.shape[0]-1)/2:
            dy = int(dy-(z.shape[0]-1))
        if dx > (z.shape[1]-1)/2:
            dx = int(dx-(z.shape[1]-1))

        # Update bounding box
        self.bbox_.shift_center(dx, dy)

        # Debugging
        res_debug = ((response - response.min()) *
                     (1/(response.max() - response.min()) * 255)).astype('uint8')
        cv2.imshow("response", res_debug)

    def __gen_kernel__(self, x: np.ndarray, x_prime: np.ndarray) -> np.ndarray:

        if (x.shape != x_prime.shape):
            raise Exception(
                f"x (shape={x.shape}) and x_prime (shape={x_prime.shape} have different shapes")

        # Calculate ||x||^2 and ||x'||^2
        x_flat = x.flatten()
        x_prime_flat = x_prime.flatten()
        mod_x_sq = (np.linalg.norm(x_flat))**2
        mod_x_prime_sq = (np.linalg.norm(x_prime_flat))**2

        # Calculate 2F^-1(F(x).F^*(x'))
        # fourier_terms = np.fft.fftshift(np.fft.ifft2(
        #     np.multiply(np.fft.fft2(x), np.fft.fft2(x_prime).conjugate())))
        fourier_terms = np.fft.ifft2(np.multiply(
            np.fft.fft2(x), np.fft.fft2(x_prime).conjugate()))

        # Sum up the fourier terms with norm of x
        fourier_terms = -2*fourier_terms+mod_x_sq + \
            mod_x_prime_sq
        fourier_terms = np.absolute(fourier_terms)  # Discard imaginary terms

        # Resulting kernel
        k = np.exp((-1.0/(self.sigma_ ** 2))*fourier_terms/(x.size))

        return k


class Bbox:
    def __init__(self, top_left_x: int = -1, top_left_y: int = -1, width: int = -1, height: int = -1):
        self.top_left_x = top_left_x
        self.top_left_y = top_left_y
        self.width = width
        self.height = height

    def center(self) -> tuple:
        return (self.top_left_x+(self.width-1)/2.0, self.top_left_y+(self.height-1)/2.0)

    def move_center(self, x_center: int, y_center: int) -> None:
        self.top_left_x = x_center-(self.width-1)/2
        self.top_left_y = y_center-(self.height-1)/2

    def shift_center(self, dx: int, dy: int) -> None:
        self.top_left_x += dx
        self.top_left_y += dy
