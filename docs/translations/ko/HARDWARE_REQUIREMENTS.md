# 최소 하드웨어 요구 사항

## 기본 사양

- **CPU:** 4코어 CPU (8코어 이상 권장)
- **RAM:** 8 GB (16 GB 이상 권장)
- **VRAM:** 0 GB (4 GB 이상 권장)
- **저장 공간:** 5 GB

## 말풍선 외 (OSB) 인페인팅 (Flux 모델)

기본 사양 외 추가 사양입니다. VRAM 수치는 근사치이며 하드웨어 사양에 따라 다를 수 있습니다.

### SDNQ 백엔드

| 모델 | VRAM | VRAM (저비디오메모리) | RAM | 저장 공간 |
| --- | --- | --- | --- | --- |
| FLUX.2 Klein 4B | +5.0 GB | +2.0 GB | +6 GB | +5.1 GB |
| FLUX.2 Klein 9B | +8.0 GB | +3.0 GB | +13 GB | +11.7 GB |
| FLUX.1 Kontext | +9.0 GB | +4.0 GB | +14 GB | +12.6 GB |

### sd.cpp 백엔드

<table>
  <thead>
    <tr>
      <th>모델</th>
      <th>양자화</th>
      <th>VRAM</th>
      <th>RAM</th>
      <th>저장 공간</th>
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
      <td rowspan="5" valign="middle"><strong>Qwen3-4B</strong><br><small>(텍스트 인코더)</small></td>
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
      <th>모델</th>
      <th>양자화</th>
      <th>VRAM</th>
      <th>RAM</th>
      <th>저장 공간</th>
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
      <td rowspan="5" valign="middle"><strong>Qwen3-8B</strong><br><small>(텍스트 인코더)</small></td>
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
      <th>모델</th>
      <th>양자화</th>
      <th>VRAM</th>
      <th>RAM</th>
      <th>저장 공간</th>
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
      <td rowspan="5" valign="middle"><strong>T5</strong><br><small>(텍스트 인코더)</small></td>
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

### Nunchaku 백엔드

| 모델 | VRAM | RAM | 저장 공간 |
| --- | --- | --- | --- |
| FLUX.1 Kontext | +6.0 GB | +11 GB | +9.7 GB |
