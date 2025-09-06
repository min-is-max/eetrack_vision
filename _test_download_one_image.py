import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge # match NumPy version
import cv2
import numpy as np

G1_TOPIC_IMAGE_LEFT = "input/left/color/000012.jpg"
G1_TOPIC_IMAGE_RIGHT = "input/right/color/000012.jpg"
G1_TOPIC_LEFT = '/zed/zed_node/left/image_rect_color'
G1_TOPIC_RIGHT = '/zed/zed_node/right/image_rect_color'

class ImageCaptureNode(Node):
    def __init__(self, topic, file):
        self.topic = topic
        self.file = file
        super().__init__('image_capture_node')
        self.subscription = self.create_subscription(
            Image,
            topic,
            self.image_callback,
            10) # check delay
        self.cam_info_subscription = self.create_subscription(
            CameraInfo,
            topic.replace("image_rect_color", "camera_info"),
            self.cam_info_callback,
            10) # check delay
        self.bridge = CvBridge()
        self.image_received = False
    
    def image_callback(self, msg):
        if not self.image_received:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                cv2.imwrite(self.file, cv_image)
                # log
                # self.get_logger().info('Image saved as ', self.file)
                self.image_received = True
                # rclpy.shutdown()
            except Exception as e:
                # log
                # self.get_logger().error(f"Error converting image: {e}")
                pass

    def cam_info_callback(self, msg):
        if "left" not in self.file:
            return
        K = np.array(msg.k).reshape(3,3)
        save_path = self.file.replace("color", "intrinsic").replace(".jpg", ".txt")
        np.savetxt(save_path, K)

def main(args=None):
    rclpy.init(args=args)
    for topic, file in zip([G1_TOPIC_LEFT, G1_TOPIC_RIGHT], [G1_TOPIC_IMAGE_LEFT, G1_TOPIC_IMAGE_RIGHT]):
        node = ImageCaptureNode(topic, file)
        for _ in range(10):
            rclpy.spin_once(node)
        node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()