#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证 faiss-gpu 是否正常工作的测试脚本
"""

import numpy as np
import faiss
import sys

def test_gpu_availability():
    """测试 GPU 是否可用"""
    print("=" * 60)
    print("1. 检查 GPU 可用性")
    print("=" * 60)
    
    try:
        num_gpus = faiss.get_num_gpus()
        print(f"✓ 检测到 {num_gpus} 个 GPU")
        
        compile_options = faiss.get_compile_options()
        if "GPU" in compile_options:
            print(f"✓ Faiss 已编译 GPU 支持")
            print(f"  编译选项: {compile_options}")
        else:
            print("✗ 警告: Faiss 未编译 GPU 支持")
            return False
            
        return True
    except Exception as e:
        print(f"✗ 错误: {e}")
        return False


def test_basic_gpu_index():
    """测试基本的 GPU 索引创建和搜索"""
    print("\n" + "=" * 60)
    print("2. 测试基本 GPU 索引 (GpuIndexFlatL2)")
    print("=" * 60)
    
    try:
        # 创建测试数据
        d = 64  # 向量维度
        nb = 10000  # 数据库大小
        nq = 100  # 查询数量
        
        np.random.seed(1234)
        xb = np.random.random((nb, d)).astype('float32')
        xq = np.random.random((nq, d)).astype('float32')
        
        # 创建 GPU 资源
        res = faiss.StandardGpuResources()
        
        # 创建 GPU 索引
        gpu_index = faiss.GpuIndexFlatL2(res, d)
        
        # 添加向量
        gpu_index.add(xb)
        print(f"✓ 成功添加 {gpu_index.ntotal} 个向量到 GPU 索引")
        
        # 搜索
        k = 10
        D, I = gpu_index.search(xq, k)
        print(f"✓ 成功搜索，返回 {I.shape[0]} 个查询的 {I.shape[1]} 个最近邻")
        print(f"  前 3 个查询的结果索引: {I[:3]}")
        
        return True
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cpu_gpu_consistency():
    """测试 CPU 和 GPU 结果一致性"""
    print("\n" + "=" * 60)
    print("3. 测试 CPU vs GPU 结果一致性")
    print("=" * 60)
    
    try:
        # 创建测试数据
        d = 64
        nb = 5000
        nq = 100
        k = 10
        
        np.random.seed(1234)
        xb = np.random.random((nb, d)).astype('float32')
        xq = np.random.random((nq, d)).astype('float32')
        
        # CPU 索引
        cpu_index = faiss.IndexFlatL2(d)
        cpu_index.add(xb)
        D_cpu, I_cpu = cpu_index.search(xq, k)
        
        # GPU 索引
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(d))
        gpu_index.add(xb)
        D_gpu, I_gpu = gpu_index.search(xq, k)
        
        # 比较结果
        indices_match = np.all(I_cpu == I_gpu)
        distances_close = np.allclose(D_cpu, D_gpu, rtol=1e-5)
        
        if indices_match and distances_close:
            print("✓ CPU 和 GPU 结果完全一致")
            print(f"  索引匹配: {indices_match}")
            print(f"  距离匹配 (rtol=1e-5): {distances_close}")
            return True
        else:
            print("✗ CPU 和 GPU 结果不一致")
            print(f"  索引匹配: {indices_match}")
            print(f"  距离匹配: {distances_close}")
            if not indices_match:
                diff_count = np.sum(I_cpu != I_gpu)
                print(f"  不匹配的索引数量: {diff_count}")
            if not distances_close:
                max_diff = np.abs(D_cpu - D_gpu).max()
                print(f"  最大距离差异: {max_diff}")
            return False
            
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ivf_gpu_index():
    """测试 IVF GPU 索引"""
    print("\n" + "=" * 60)
    print("4. 测试 IVF GPU 索引 (GpuIndexIVFFlat)")
    print("=" * 60)
    
    try:
        d = 64
        nb = 10000
        nq = 100
        nlist = 100
        
        np.random.seed(1234)
        xb = np.random.random((nb, d)).astype('float32')
        xq = np.random.random((nq, d)).astype('float32')
        
        # 创建 IVF GPU 索引
        res = faiss.StandardGpuResources()
        gpu_index = faiss.GpuIndexIVFFlat(res, d, nlist, faiss.METRIC_L2)
        
        # 训练索引
        print("  训练索引...")
        gpu_index.train(xb)
        print(f"✓ 索引已训练: {gpu_index.is_trained}")
        
        # 添加向量
        gpu_index.add(xb)
        print(f"✓ 成功添加 {gpu_index.ntotal} 个向量")
        
        # 搜索
        gpu_index.nprobe = 10
        k = 10
        D, I = gpu_index.search(xq, k)
        print(f"✓ 成功搜索，返回 {I.shape[0]} 个查询的 {I.shape[1]} 个最近邻")
        
        return True
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("Faiss GPU 验证测试")
    print("=" * 60)
    
    results = []
    
    # 运行所有测试
    results.append(("GPU 可用性", test_gpu_availability()))
    results.append(("基本 GPU 索引", test_basic_gpu_index()))
    results.append(("CPU/GPU 一致性", test_cpu_gpu_consistency()))
    results.append(("IVF GPU 索引", test_ivf_gpu_index()))
    
    # 打印总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 所有测试通过！Faiss GPU 工作正常。")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查错误信息。")
        return 1


if __name__ == "__main__":
    sys.exit(main())

