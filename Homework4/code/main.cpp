#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

std::vector<cv::Point2f> control_points;

void mouse_handler(int event, int x, int y, int flags, void *userdata) 
{
    if (event == cv::EVENT_LBUTTONDOWN && control_points.size() < 4) 
    {
        std::cout << "Left button of the mouse is clicked - position (" << x << ", "
        << y << ")" << '\n';
        control_points.emplace_back(x, y);
    }     
}

void naive_bezier(const std::vector<cv::Point2f> &points, cv::Mat &window) 
{
    auto &p_0 = points[0];
    auto &p_1 = points[1];
    auto &p_2 = points[2];
    auto &p_3 = points[3];

    for (double t = 0.0; t <= 1.0; t += 0.001) 
    {
        auto point = std::pow(1 - t, 3) * p_0 + 3 * t * std::pow(1 - t, 2) * p_1 +
                 3 * std::pow(t, 2) * (1 - t) * p_2 + std::pow(t, 3) * p_3;

        window.at<cv::Vec3b>(point.y, point.x)[2] = 255;
    }
}

cv::Point2f recursive_bezier(const std::vector<cv::Point2f> &control_points, float t) 
{
    // TODO: Implement de Casteljau's algorithm
    // t = std::clamp(t, 0.0f, 1.0f);
    if(t < 0.0f){
        t = 0.0f;
    } else if(t > 1.0f){
        t = 1.0f;
    }
    if(control_points.size() == 1){
        return control_points[0];
    }

    std::vector<cv::Point2f> next;
    next.reserve(control_points.size() - 1);

    for(size_t i = 0; i < control_points.size() - 1; ++i){
        cv::Point2f p = (1.0f - t) * control_points[i] + t * control_points[i + 1];
        next.push_back(p);
    }

    return recursive_bezier(next, t);

    // return cv::Point2f();

}

void bezier(const std::vector<cv::Point2f> &control_points, cv::Mat &window) 
{
    // TODO: Iterate through all t = 0 to t = 1 with small steps, and call de Casteljau's 
    // recursive Bezier algorithm.
    if(control_points.empty()){
        return;
    }

    const int height = window.rows;
    const int width = window.cols;

    const float step = 0.001f;
    cv::Point prev_pt(-1, -1);

    for(float t = 0.0f; t < 1.0f; t += step){
        cv::Point2f p = recursive_bezier(control_points, t);

        int x = static_cast<int>(std::lround(p.x));
        int y = static_cast<int>(std::lround(p.y));

        if(x >= 0 && x < width && y >= 0 && y < height){
            cv::circle(window, cv::Point(x, y), 1, cv::Scalar(0, 255, 0), -1);
        }

        if(prev_pt.x >= 0){
            if((prev_pt.x >= 0 && prev_pt.x < width && prev_pt.y >= 0 && prev_pt.y < height) &&
               (x >= 0 && x < width && y >= 0 && y < height)){
                cv::line(window, prev_pt, cv::Point(x, y), cv::Scalar(0, 255, 0), 1);
            }
        }
        prev_pt = cv::Point(x, y);
    }
    if(prev_pt.x < 0 || prev_pt.x >= width || prev_pt.y < 0 || prev_pt.y >= height){
        cv::Point2f p1 = recursive_bezier(control_points, 1.0f);
        int x1 = static_cast<int>(std::lround(p1.x));
        int y1 = static_cast<int>(std::lround(p1.y));
        if(x1 >= 0 && x1 < width && y1 >= 0 && y1 < height){
            cv::circle(window, cv::Point(x1, y1), 1, cv::Scalar(0, 255, 0), -1);
        }
    }

}

int main() 
{
    cv::Mat window = cv::Mat(700, 700, CV_8UC3, cv::Scalar(0));
    cv::cvtColor(window, window, cv::COLOR_BGR2RGB);
    cv::namedWindow("Bezier Curve", cv::WINDOW_AUTOSIZE);

    cv::setMouseCallback("Bezier Curve", mouse_handler, nullptr);

    int key = -1;
    while (key != 27) 
    {
        for (auto &point : control_points) 
        {
            cv::circle(window, point, 3, {255, 255, 255}, 3);
        }

        if (control_points.size() == 4) 
        {
            bezier(control_points, window);
            naive_bezier(control_points, window);
            

            cv::imshow("Bezier Curve", window);
            cv::imwrite("my_bezier_curve.png", window);
            key = cv::waitKey(0);

            return 0;
        }

        cv::imshow("Bezier Curve", window);
        key = cv::waitKey(20);
    }

return 0;
}
