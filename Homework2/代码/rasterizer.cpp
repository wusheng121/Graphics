// clang-format off
//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include <vector>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>


rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions)
{
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices)
{
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols)
{
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}


static bool insideTriangle(float x, float y, const Vector3f* _v)
{   
    // TODO : Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]
    //将点坐标转为向量
    Vector2f v0 (_v[0].x(), _v[0].y());
    Vector2f v1 (_v[1].x(), _v[1].y());
    Vector2f v2 (_v[2].x(), _v[2].y());
    Vector2f p((float)x, (float)y);
    //计算叉积
    Vector2f v0v1 = v1 - v0;
    Vector2f v0p = p - v0;
    float cross0 = v0v1.x() * v0p.y() - v0v1.y() * v0p.x();

    Vector2f v1v2 = v2 - v1;
    Vector2f v1p = p - v1;
    float cross1 = v1v2.x() * v1p.y() - v1v2.y() * v1p.x();

    Vector2f v2v0 = v0 - v2;
    Vector2f v2p = p - v2;
    float cross2 = v2v0.x() * v2p.y() - v2v0.y() * v2p.x();

    //判断符号是否相同,同正同负或存在0则说明在三角形内部
    bool has_positive = (cross0 > 0 || cross1 > 0 || cross2 > 0);
    bool has_negative = (cross0 < 0 || cross1 < 0 || cross2 < 0);
    return !(has_positive && has_negative);

}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f* v)
{
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    return {c1,c2,c3};
}

void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type)
{
    auto& buf = pos_buf[pos_buffer.pos_id];
    auto& ind = ind_buf[ind_buffer.ind_id];
    auto& col = col_buf[col_buffer.col_id];

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    for (auto& i : ind)
    {
        Triangle t;
        Eigen::Vector4f v[] = {
                mvp * to_vec4(buf[i[0]], 1.0f),
                mvp * to_vec4(buf[i[1]], 1.0f),
                mvp * to_vec4(buf[i[2]], 1.0f)
        };
        //Homogeneous division
        for (auto& vec : v) {
            vec /= vec.w();
        }
        //Viewport transformation
        for (auto & vert : v)
        {
            vert.x() = 0.5*width*(vert.x()+1.0);
            vert.y() = 0.5*height*(vert.y()+1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i)
        {
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
        }

        auto col_x = col[i[0]];
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);

        rasterize_triangle(t);
    }
}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t) {
    auto v = t.toVector4();
    
    // TODO : Find out the bounding box of current triangle.
    // iterate through the pixel and find if the current pixel is inside the triangle

    // If so, use the following code to get the interpolated z value.
    // auto[alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
    // float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
    // float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
    // z_interpolated *= w_reciprocal;

    // TODO : set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.

    //创建bounding box
    int x_min = std::min({v[0].x(), v[1].x(), v[2].x()});
    int x_max = std::max({v[0].x(), v[1].x(), v[2].x()});
    int y_min = std::min({v[0].y(), v[1].y(), v[2].y()});
    int y_max = std::max({v[0].y(), v[1].y(), v[2].y()});

    const std::vector<std::pair<float, float>> samples = {
        {0.25f, 0.25f}, {0.75f, 0.25f},
        {0.25f, 0.75f}, {0.75f, 0.75f}
    };

    //遍历bounding box内的像素点
    for(int x = x_min; x <= x_max; x++){
        for(int y = y_min; y <= y_max; y++){
            //判断像素点是否在三角形内部
            const Vector3f vectices[3] = {
                {v[0].x(), v[0].y(), v[0].z()},
                {v[1].x(), v[1].y(), v[1].z()},
                {v[2].x(), v[2].y(), v[2].z()}
            };
            int sample_count = 0;
            float min_z = std::numeric_limits<float>::infinity();
            //遍历每个样本点
            for (const auto& sample : samples) {
                float sample_x = x + sample.first;
                float sample_y = y + sample.second;

                if (insideTriangle(sample_x, sample_y, vectices)) {
                    sample_count++;

                    auto [alpha, beta, gamma] = computeBarycentric2D(sample_x, sample_y, t.v);
                    float w_reciprocal = 1.0f / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                    float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                    z_interpolated *= w_reciprocal;

                    if (z_interpolated < min_z) {
                        min_z = z_interpolated;
                    }
                }
            }          
            //更新像素颜色
            if (sample_count > 0 && sample_count < 4) {
                std::cout << "Edge pixel at (" << x << "," << y << "), samples = " << sample_count << std::endl;
            }
            if(sample_count > 0) {
                //计算z_buffer
                int index = get_index(x, y);
                if (min_z < depth_buf[index]) {
                    depth_buf[index] = min_z;
                    //设置像素颜色
                    Eigen::Vector3f color = t.getColor() * (sample_count / 4.0f);
                    set_pixel(Eigen::Vector3f(x, y, 0), color);
                }
            }
            // if(insideTriangle(x + 0.5f, y + 0.5f, vectices)){
            //     //计算插值深度
            //     auto[alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
            //     float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
            //     float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
            //     z_interpolated *= w_reciprocal;

            //     //计算z_buffer
            //     int index = get_index(x, y);
            //     if(z_interpolated < depth_buf[index]){
            //         depth_buf[index] = z_interpolated;
            //         //设置像素颜色
            //         set_pixel(Vector3f(x, y, 0), t.getColor());
            //     }
            // }
        }
    }
}

void rst::rasterizer::set_model(const Eigen::Matrix4f& m)
{
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f& v)
{
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f& p)
{
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height-1-y)*width + x;
}

void rst::rasterizer::set_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color)
{
    //old index: auto ind = point.y() + point.x() * width;
    auto ind = (height-1-point.y())*width + point.x();
    frame_buf[ind] = color;

}

// clang-format on