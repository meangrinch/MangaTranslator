# 最低硬件运行要求

## 基础要求

- **CPU:** 4 核 CPU (推荐 8 核及以上)
- **内存 (RAM):** 8 GB (推荐 16 GB 及以上)
- **显存 (VRAM):** 0 GB (推荐 4 GB 及以上)
- **存储空间:** 5 GB

## 对话框外 (OSB) 擦除重绘 (Flux 模型)

基础要求之上的附加要求。显存 (VRAM) 数据为近似值，且取决于具体硬件。

### SDNQ 后端

| 模型 | 显存 (VRAM) | 显存 (低显存模式) | 内存 (RAM) | 存储空间 |
| --- | --- | --- | --- | --- |
| FLUX.2 Klein 4B | +5.0 GB | +2.0 GB | +6 GB | +5.1 GB |
| FLUX.2 Klein 9B | +8.0 GB | +3.0 GB | +13 GB | +11.7 GB |
| FLUX.1 Kontext | +9.0 GB | +4.0 GB | +14 GB | +12.6 GB |

### sd.cpp 后端

<table>
  <thead>
    <tr>
      <th>模型</th>
      <th>量化</th>
      <th>显存 (VRAM)</th>
      <th>内存 (RAM)</th>
      <th>存储空间</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5" valign="middle"><strong>FLUX.2 Klein 4B</strong></td>
      <td>Q8_0</td>
      <td>+7.3 GB</td>
      <td>+5.5 GB</td>
      <td>+4.0 GB</td>
    </tr>
    <tr>
      <td>Q6_K</td>
      <td>+6.5 GB</td>
      <td>+4.7 GB</td>
      <td>+3.2 GB</td>
    </tr>
    <tr>
      <td>Q5_K_M</td>
      <td>+6.2 GB</td>
      <td>+4.4 GB</td>
      <td>+2.9 GB</td>
    </tr>
    <tr>
      <td>Q4_K_M</td>
      <td>+5.7 GB</td>
      <td>+3.9 GB</td>
      <td>+2.4 GB</td>
    </tr>
    <tr>
      <td>Q3_K_M</td>
      <td>+5.3 GB</td>
      <td>+3.5 GB</td>
      <td>+2.0 GB</td>
    </tr>
    <tr>
      <td rowspan="5" valign="middle"><strong>Qwen3-4B</strong><br><small>(文本编码器)</small></td>
      <td>Q8_0</td>
      <td>-</td>
      <td>+4.0 GB</td>
      <td>+4.0 GB</td>
    </tr>
    <tr>
      <td>Q6_K_XL</td>
      <td>-</td>
      <td>+3.4 GB</td>
      <td>+3.4 GB</td>
    </tr>
    <tr>
      <td>Q5_K_XL</td>
      <td>-</td>
      <td>+2.7 GB</td>
      <td>+2.7 GB</td>
    </tr>
    <tr>
      <td>Q4_K_XL</td>
      <td>-</td>
      <td>+2.4 GB</td>
      <td>+2.4 GB</td>
    </tr>
    <tr>
      <td>Q3_K_XL</td>
      <td>-</td>
      <td>+2.0 GB</td>
      <td>+2.0 GB</td>
    </tr>
  </tbody>
</table>

---

<table>
  <thead>
    <tr>
      <th>模型</th>
      <th>量化</th>
      <th>显存 (VRAM)</th>
      <th>内存 (RAM)</th>
      <th>存储空间</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5" valign="middle"><strong>FLUX.2 Klein 9B</strong></td>
      <td>Q8_0</td>
      <td>+12.6 GB</td>
      <td>+10.8 GB</td>
      <td>+9.3 GB</td>
    </tr>
    <tr>
      <td>Q6_K</td>
      <td>+10.6 GB</td>
      <td>+8.8 GB</td>
      <td>+7.3 GB</td>
    </tr>
    <tr>
      <td>Q5_K_M</td>
      <td>+9.8 GB</td>
      <td>+8.0 GB</td>
      <td>+6.5 GB</td>
    </tr>
    <tr>
      <td>Q4_K_M</td>
      <td>+8.8 GB</td>
      <td>+7.0 GB</td>
      <td>+5.5 GB</td>
    </tr>
    <tr>
      <td>Q3_K_M</td>
      <td>+7.7 GB</td>
      <td>+5.9 GB</td>
      <td>+4.4 GB</td>
    </tr>
    <tr>
      <td rowspan="5" valign="middle"><strong>Qwen3-8B</strong><br><small>(文本编码器)</small></td>
      <td>Q8_0</td>
      <td>-</td>
      <td>+8.1 GB</td>
      <td>+8.1 GB</td>
    </tr>
    <tr>
      <td>Q6_K_XL</td>
      <td>-</td>
      <td>+7.0 GB</td>
      <td>+7.0 GB</td>
    </tr>
    <tr>
      <td>Q5_K_XL</td>
      <td>-</td>
      <td>+5.5 GB</td>
      <td>+5.5 GB</td>
    </tr>
    <tr>
      <td>Q4_K_XL</td>
      <td>-</td>
      <td>+4.8 GB</td>
      <td>+4.8 GB</td>
    </tr>
    <tr>
      <td>Q3_K_XL</td>
      <td>-</td>
      <td>+4.0 GB</td>
      <td>+4.0 GB</td>
    </tr>
  </tbody>
</table>

---

<table>
  <thead>
    <tr>
      <th>模型</th>
      <th>量化</th>
      <th>显存 (VRAM)</th>
      <th>内存 (RAM)</th>
      <th>存储空间</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5" valign="middle"><strong>FLUX.1 Kontext</strong></td>
      <td>Q8_0</td>
      <td>+15.1 GB</td>
      <td>+13.3 GB</td>
      <td>+11.8 GB</td>
    </tr>
    <tr>
      <td>Q6_K</td>
      <td>+12.5 GB</td>
      <td>+10.7 GB</td>
      <td>+9.2 GB</td>
    </tr>
    <tr>
      <td>Q5_K_M</td>
      <td>+11.1 GB</td>
      <td>+9.3 GB</td>
      <td>+7.8 GB</td>
    </tr>
    <tr>
      <td>Q4_K_M</td>
      <td>+9.8 GB</td>
      <td>+8.0 GB</td>
      <td>+6.5 GB</td>
    </tr>
    <tr>
      <td>Q3_K_M</td>
      <td>+8.3 GB</td>
      <td>+6.5 GB</td>
      <td>+5.0 GB</td>
    </tr>
    <tr>
      <td rowspan="5" valign="middle"><strong>T5</strong><br><small>(文本编码器)</small></td>
      <td>Q8_0</td>
      <td>-</td>
      <td>+4.7 GB</td>
      <td>+4.7 GB</td>
    </tr>
    <tr>
      <td>Q6_K</td>
      <td>-</td>
      <td>+3.6 GB</td>
      <td>+3.6 GB</td>
    </tr>
    <tr>
      <td>Q5_K_M</td>
      <td>-</td>
      <td>+3.2 GB</td>
      <td>+3.2 GB</td>
    </tr>
    <tr>
      <td>Q4_K_M</td>
      <td>-</td>
      <td>+2.7 GB</td>
      <td>+2.7 GB</td>
    </tr>
    <tr>
      <td>Q3_K_M</td>
      <td>-</td>
      <td>+2.1 GB</td>
      <td>+2.1 GB</td>
    </tr>
  </tbody>
</table>

### Nunchaku 后端

| 模型 | 显存 (VRAM) | 内存 (RAM) | 存储空间 |
| --- | --- | --- | --- |
| FLUX.1 Kontext | +6.0 GB | +11 GB | +9.7 GB |
