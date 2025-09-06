import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np

G1_INPUT_IMAGE_LEFT = "input/left/color/000012.jpg"
G1_INPUT_IMAGE_RIGHT = "input/right/color/000012.jpg"
G1_INPUT_IMAGE_INTRINSIC = "input/left/intrinsic/000012.txt" # intrinsic from left

G1_TOPIC_LEFT_COLOR = "/zed/zed_node/left/image_rect_color"
G1_TOPIC_RIGHT_COLOR = "/zed/zed_node/right/image_rect_color"
G1_TOPIC_CAMERA_INFO = "/zed/zed_node/left/camera_info" # intrinsic from left


class ZedCaptureNode(Node):
    def __init__(self, topic, file):
        self.topic = topic
        self.file = file
        topic_type = topic.split('/')[-1]
        topic_class = Image if topic_type == "image_rect_color" else CameraInfo
        super().__init__("zed_capture_node")
        self.subscription = self.create_subscription(
            topic_class,
            topic,
            self.zed_callback,
            10)
        self.bridge = CvBridge()
        self.received = False

    def zed_callback(self, msg):
        if not self.received:
            try:
                if self.topic.split('/')[-1] == "image_rect_color":
                    cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                    cv2.imwrite(self.file, cv_image)
                    self.received = True
                    self.get_logger().info(f"Zed image saved as {self.file}")
                elif self.topic.split('/')[-1] == "camera_info":
                    K = np.array(msg.k).reshape(3,3)
                    np.savetxt(self.file, K)
                    self.received = True
                    self.get_logger().info(f"Zed topic Message captured: {self.topic}")
                else:
                    # no matching zed topics found
                    self.received = True # prevent requesting for callback
                    self.get_logger().info(f"Unknown topic: {self.topic}")
                    
            except Exception as e:
                self.get_logger().error(f"Error capturing zed topics: {e}")

def main(args=None):
    rclpy.init(args=args)
    for topic, file, in zip([G1_TOPIC_LEFT_COLOR, G1_TOPIC_RIGHT_COLOR, G1_TOPIC_CAMERA_INFO], [G1_INPUT_IMAGE_LEFT, G1_INPUT_IMAGE_RIGHT, G1_INPUT_IMAGE_INTRINSIC]):
        node = ZedCaptureNode(topic, file)
        for _ in range(10):
            rclpy.spin_once(node)
        node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
