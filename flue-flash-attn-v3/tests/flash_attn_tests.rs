use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor, D};
use flue_flash_attn_v3;
use rstest::rstest;

fn to_vec3_round(t: Tensor, digits: i32) -> Result<Vec<Vec<Vec<f32>>>> {
    let b = 10f32.powi(digits);
    let t = t.to_vec3::<f32>()?;
    let t = t
        .iter()
        .map(|t| {
            t.iter()
                .map(|t| t.iter().map(|t| f32::round(t * b) / b).collect())
                .collect()
        })
        .collect();
    Ok(t)
}

fn fa_acausal(q: &Tensor, k: &Tensor, v: &Tensor, softmax_scale: f32) -> Result<Tensor> {
    let in_dtype = q.dtype();
    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;
    let att = (q.matmul(&k.t()?)? * softmax_scale as f64)?;
    let att = candle_nn::ops::softmax(&att, D::Minus1)?;
    // Convert to contiguous as matmul doesn't support strided vs for now.
    let output = att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?;
    Ok(output)
}

#[test]
fn flash_attn_acausal() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let q = Tensor::arange(0u32, 3 * 2 * 64, &device)?
        .to_dtype(DType::F16)?
        .reshape((1, 3, 2, 64))?;
    let k = (&q / 400.)?;
    let v = (&q / 500.)?;
    let q = (&q / 300.)?;

    let ys1 = fa_acausal(&q, &k, &v, 0.5)?;
    let ys1 = ys1.i(0)?.to_dtype(DType::F32)?;
    let ys2 = {
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        flue_flash_attn_v3::flash_attn(&q, &k, &v, 0.5, false, false)?.transpose(1, 2)?
    };
    let ys2 = ys2.i(0)?.to_dtype(DType::F32)?;
    let diff = ys1.sub(&ys2)?.abs()?.flatten_all()?.max(0)?;

    assert_eq!(ys2.dims(), &[3, 2, 64]);
    assert_eq!(
        to_vec3_round(ys2, 4)?,
        &[
            [
                [
                    0.0808, 0.0828, 0.0848, 0.0869, 0.0889, 0.0908, 0.0928, 0.0948, 0.0969, 0.0989,
                    0.1008, 0.1028, 0.1049, 0.1069, 0.1088, 0.1108, 0.1129, 0.1149, 0.1168, 0.1188,
                    0.1208, 0.1229, 0.1249, 0.1268, 0.1288, 0.1309, 0.1328, 0.1349, 0.1368, 0.1388,
                    0.1409, 0.1428, 0.1449, 0.1469, 0.1488, 0.1509, 0.1528, 0.1548, 0.1569, 0.1588,
                    0.1609, 0.1628, 0.1648, 0.1669, 0.1688, 0.1709, 0.1729, 0.1748, 0.1769, 0.1788,
                    0.1809, 0.1829, 0.1848, 0.1869, 0.1888, 0.1908, 0.1929, 0.1948, 0.1969, 0.1989,
                    0.2008, 0.2029, 0.205, 0.2069
                ],
                [
                    0.1071, 0.1091, 0.1111, 0.113, 0.1151, 0.1171, 0.1191, 0.1211, 0.123, 0.1251,
                    0.1271, 0.129, 0.1311, 0.1331, 0.135, 0.1371, 0.139, 0.1411, 0.1431, 0.145,
                    0.1471, 0.149, 0.1511, 0.1531, 0.155, 0.1571, 0.1591, 0.1611, 0.1631, 0.165,
                    0.1671, 0.1691, 0.1711, 0.1731, 0.175, 0.1771, 0.1791, 0.181, 0.1831, 0.1851,
                    0.1871, 0.1891, 0.191, 0.1931, 0.1951, 0.1971, 0.1991, 0.201, 0.2031, 0.2051,
                    0.2072, 0.2091, 0.2111, 0.2131, 0.2151, 0.217, 0.2191, 0.2211, 0.2231, 0.2251,
                    0.2271, 0.229, 0.2312, 0.2332
                ]
            ],
            [
                [
                    0.3765, 0.3784, 0.3804, 0.3823, 0.3843, 0.3862, 0.3884, 0.3904, 0.3923, 0.3943,
                    0.3962, 0.3984, 0.4004, 0.4023, 0.4043, 0.4063, 0.4084, 0.4104, 0.4124, 0.4143,
                    0.4163, 0.4185, 0.4204, 0.4224, 0.4243, 0.4263, 0.4285, 0.4304, 0.4324, 0.4343,
                    0.4363, 0.4385, 0.4404, 0.4424, 0.4443, 0.4463, 0.4485, 0.4504, 0.4524, 0.4543,
                    0.4563, 0.4585, 0.4604, 0.4624, 0.4644, 0.4663, 0.4683, 0.4705, 0.4724, 0.4744,
                    0.4763, 0.4783, 0.4805, 0.4824, 0.4844, 0.4863, 0.4883, 0.4905, 0.4922, 0.4946,
                    0.4966, 0.4985, 0.5005, 0.5024
                ],
                [
                    0.3816, 0.3835, 0.3855, 0.3875, 0.3894, 0.3914, 0.3936, 0.3955, 0.3975, 0.3994,
                    0.4014, 0.4036, 0.4055, 0.4075, 0.4094, 0.4114, 0.4136, 0.4155, 0.4175, 0.4194,
                    0.4214, 0.4236, 0.4255, 0.4275, 0.4294, 0.4314, 0.4336, 0.4355, 0.4375, 0.4395,
                    0.4414, 0.4436, 0.4456, 0.4475, 0.4495, 0.4514, 0.4536, 0.4556, 0.4575, 0.4595,
                    0.4614, 0.4636, 0.4656, 0.4675, 0.4695, 0.4714, 0.4734, 0.4756, 0.4775, 0.4795,
                    0.4814, 0.4834, 0.4856, 0.4875, 0.4895, 0.4915, 0.4934, 0.4956, 0.4973, 0.4998,
                    0.5015, 0.5034, 0.5054, 0.5073
                ]
            ],
            [
                [
                    0.6392, 0.6411, 0.6431, 0.6455, 0.6475, 0.6494, 0.6514, 0.6533, 0.6553, 0.6572,
                    0.6592, 0.6611, 0.6631, 0.6655, 0.6675, 0.6694, 0.6714, 0.6733, 0.6753, 0.6772,
                    0.6792, 0.6812, 0.6831, 0.6851, 0.6875, 0.6895, 0.6914, 0.6934, 0.6953, 0.6973,
                    0.6992, 0.7012, 0.7031, 0.7051, 0.7075, 0.7095, 0.7114, 0.7134, 0.7153, 0.7173,
                    0.7192, 0.7212, 0.7231, 0.7251, 0.7275, 0.7295, 0.7314, 0.7334, 0.7354, 0.7373,
                    0.7393, 0.7412, 0.7432, 0.7451, 0.7476, 0.7495, 0.7515, 0.7534, 0.7554, 0.7573,
                    0.7593, 0.7612, 0.7632, 0.7651
                ],
                [
                    0.6396, 0.6416, 0.6436, 0.646, 0.6479, 0.6499, 0.6519, 0.6538, 0.6558, 0.6577,
                    0.6597, 0.6616, 0.6636, 0.666, 0.668, 0.6699, 0.6719, 0.6738, 0.6758, 0.6777,
                    0.6797, 0.6816, 0.6836, 0.6855, 0.688, 0.6899, 0.6919, 0.6938, 0.6958, 0.6978,
                    0.6997, 0.7017, 0.7036, 0.7056, 0.708, 0.71, 0.7119, 0.7139, 0.7158, 0.7178,
                    0.7197, 0.7217, 0.7236, 0.7256, 0.728, 0.73, 0.7319, 0.7339, 0.7358, 0.7378,
                    0.7397, 0.7417, 0.7437, 0.7456, 0.748, 0.75, 0.752, 0.7539, 0.7559, 0.7578,
                    0.7598, 0.7617, 0.7637, 0.7656
                ]
            ]
        ]
    );
    assert!(diff.to_vec0::<f32>()?.abs() < 1e-5);
    Ok(())
}

#[test]
fn flash_attn_acausal_gqa() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let n_h = 4usize;
    let n_h_k = 1usize;

    let q = Tensor::arange(0u32, (n_h * 2 * 64) as u32, &device)?
        .to_dtype(DType::F16)?
        .reshape((1, n_h, 2, 64))?;
    let gqa = q.clone().i((.., ..n_h_k, .., ..))?;
    assert_eq!(gqa.dims(), &[1, n_h_k, 2, 64]);

    let q = (q.clone() / 1000.)?;
    let k_gqa = (&gqa / 400.)?;
    let v_gqa = (&gqa / 500.)?;

    // let gqa_repeat = gqa.repeat((1, (n_h / n_h_k) as usize, 1, 1))?;
    // assert_eq!(gqa_repeat.dims(), &[1, n_h, 2, 64]);
    // let k = (&gqa_repeat / 400.)?;
    // let v = (&gqa_repeat / 500.)?;

    // let ys1 = fa_acausal(&q, &k, &v, 0.5)?;
    // let ys1 = ys1.i(0)?.to_dtype(DType::F32)?;
    // assert_eq!(ys1.dims(), &[n_h, 2, 64]);

    let ys2 = {
        let q = q.transpose(1, 2)?;
        let k_gqa = k_gqa.transpose(1, 2)?;
        let v_gqa = v_gqa.transpose(1, 2)?;
        flue_flash_attn_v3::flash_attn(&q, &k_gqa, &v_gqa, 0.125, false, true)?.transpose(1, 2)?
    };
    let ys2 = ys2.i(0)?.to_dtype(DType::F32)?;
    assert_eq!(ys2.dims(), &[n_h, 2, 64]);

    assert_eq!(
        to_vec3_round(ys2.clone(), 4)?,
        &[
            [
                [
                    0.0653, 0.0673, 0.0693, 0.0713, 0.0734, 0.0753, 0.0773, 0.0793, 0.0813, 0.0834,
                    0.0853, 0.0873, 0.0894, 0.0913, 0.0933, 0.0953, 0.0973, 0.0994, 0.1013, 0.1033,
                    0.1053, 0.1073, 0.1094, 0.1113, 0.1133, 0.1154, 0.1173, 0.1194, 0.1213, 0.1233,
                    0.1254, 0.1273, 0.1294, 0.1313, 0.1333, 0.1354, 0.1373, 0.1393, 0.1414, 0.1433,
                    0.1454, 0.1473, 0.1493, 0.1514, 0.1533, 0.1554, 0.1573, 0.1593, 0.1614, 0.1633,
                    0.1654, 0.1674, 0.1693, 0.1714, 0.1733, 0.1753, 0.1774, 0.1793, 0.1814, 0.1833,
                    0.1853, 0.1874, 0.1895, 0.1914
                ],
                [
                    0.0679, 0.0699, 0.072, 0.0739, 0.076, 0.0779, 0.0799, 0.082, 0.0839, 0.086,
                    0.088, 0.0899, 0.092, 0.0939, 0.0959, 0.098, 0.0999, 0.102, 0.1039, 0.106,
                    0.108, 0.1099, 0.112, 0.114, 0.1159, 0.118, 0.1199, 0.122, 0.124, 0.126,
                    0.1279, 0.13, 0.132, 0.134, 0.136, 0.1379, 0.14, 0.142, 0.144, 0.146, 0.1479,
                    0.1499, 0.152, 0.1539, 0.1559, 0.158, 0.1599, 0.162, 0.1639, 0.1659, 0.168,
                    0.1699, 0.172, 0.174, 0.1759, 0.178, 0.1799, 0.182, 0.184, 0.1859, 0.188,
                    0.1899, 0.192, 0.194
                ]
            ],
            [
                [
                    0.0706, 0.0725, 0.0746, 0.0765, 0.0786, 0.0806, 0.0825, 0.0846, 0.0865, 0.0886,
                    0.0906, 0.0925, 0.0946, 0.0966, 0.0985, 0.1006, 0.1025, 0.1046, 0.1066, 0.1085,
                    0.1106, 0.1125, 0.1146, 0.1166, 0.1185, 0.1206, 0.1226, 0.1246, 0.1266, 0.1285,
                    0.1306, 0.1326, 0.1346, 0.1366, 0.1385, 0.1406, 0.1426, 0.1445, 0.1466, 0.1486,
                    0.1506, 0.1526, 0.1545, 0.1566, 0.1586, 0.1606, 0.1626, 0.1646, 0.1666, 0.1686,
                    0.1707, 0.1726, 0.1746, 0.1766, 0.1786, 0.1805, 0.1826, 0.1846, 0.1866, 0.1886,
                    0.1906, 0.1925, 0.1947, 0.1967
                ],
                [
                    0.0731, 0.0751, 0.0771, 0.0791, 0.0812, 0.0831, 0.0851, 0.0872, 0.0891, 0.0912,
                    0.0931, 0.0951, 0.0972, 0.0991, 0.1011, 0.1031, 0.1051, 0.1072, 0.1091, 0.1111,
                    0.1132, 0.1151, 0.1172, 0.1191, 0.1212, 0.1232, 0.1251, 0.1272, 0.1292, 0.1311,
                    0.1332, 0.1351, 0.1372, 0.1392, 0.1411, 0.1432, 0.1451, 0.1471, 0.1492, 0.1511,
                    0.1532, 0.1552, 0.1571, 0.1592, 0.1611, 0.1632, 0.1652, 0.1671, 0.1692, 0.1711,
                    0.1732, 0.1752, 0.1771, 0.1792, 0.1812, 0.1831, 0.1852, 0.1871, 0.1892, 0.1912,
                    0.1931, 0.1951, 0.1973, 0.1992
                ]
            ],
            [
                [
                    0.0757, 0.0776, 0.0797, 0.0817, 0.0837, 0.0857, 0.0876, 0.0897, 0.0917, 0.0938,
                    0.0957, 0.0977, 0.0997, 0.1017, 0.1036, 0.1057, 0.1077, 0.1097, 0.1117, 0.1136,
                    0.1157, 0.1177, 0.1198, 0.1217, 0.1237, 0.1257, 0.1277, 0.1298, 0.1317, 0.1337,
                    0.1357, 0.1377, 0.1398, 0.1417, 0.1437, 0.1458, 0.1477, 0.1497, 0.1517, 0.1537,
                    0.1558, 0.1577, 0.1597, 0.1617, 0.1637, 0.1658, 0.1677, 0.1697, 0.1718, 0.1737,
                    0.1758, 0.1777, 0.1797, 0.1818, 0.1837, 0.1857, 0.1877, 0.1897, 0.1918, 0.1937,
                    0.1957, 0.1976, 0.1998, 0.2018
                ],
                [
                    0.0782, 0.0802, 0.0822, 0.0842, 0.0862, 0.0882, 0.0902, 0.0922, 0.0942, 0.0963,
                    0.0982, 0.1002, 0.1022, 0.1042, 0.1062, 0.1082, 0.1102, 0.1122, 0.1142, 0.1162,
                    0.1182, 0.1202, 0.1223, 0.1242, 0.1262, 0.1283, 0.1302, 0.1322, 0.1343, 0.1362,
                    0.1383, 0.1403, 0.1422, 0.1443, 0.1462, 0.1482, 0.1503, 0.1522, 0.1543, 0.1563,
                    0.1582, 0.1603, 0.1622, 0.1643, 0.1663, 0.1682, 0.1703, 0.1722, 0.1743, 0.1763,
                    0.1782, 0.1803, 0.1823, 0.1843, 0.1863, 0.1882, 0.1903, 0.1923, 0.1943, 0.1963,
                    0.1982, 0.2002, 0.2023, 0.2043
                ]
            ],
            [
                [
                    0.0807, 0.0826, 0.0847, 0.0867, 0.0887, 0.0907, 0.0927, 0.0947, 0.0967, 0.0987,
                    0.1007, 0.1027, 0.1047, 0.1067, 0.1086, 0.1107, 0.1127, 0.1147, 0.1167, 0.1187,
                    0.1207, 0.1227, 0.1247, 0.1267, 0.1287, 0.1307, 0.1327, 0.1348, 0.1367, 0.1387,
                    0.1407, 0.1427, 0.1448, 0.1467, 0.1487, 0.1508, 0.1527, 0.1547, 0.1567, 0.1587,
                    0.1608, 0.1627, 0.1647, 0.1667, 0.1687, 0.1708, 0.1727, 0.1747, 0.1768, 0.1787,
                    0.1808, 0.1827, 0.1847, 0.1868, 0.1887, 0.1907, 0.1927, 0.1947, 0.1968, 0.1987,
                    0.2007, 0.2026, 0.2048, 0.2068
                ],
                [
                    0.0831, 0.0851, 0.0871, 0.0891, 0.0911, 0.0931, 0.0951, 0.0971, 0.0991, 0.1011,
                    0.1031, 0.1051, 0.1071, 0.1091, 0.1111, 0.1131, 0.1151, 0.1171, 0.1191, 0.1211,
                    0.1231, 0.1251, 0.1271, 0.1292, 0.1311, 0.1332, 0.1351, 0.1371, 0.1392, 0.1411,
                    0.1432, 0.1451, 0.1471, 0.1492, 0.1511, 0.1531, 0.1552, 0.1571, 0.1592, 0.1611,
                    0.1631, 0.1652, 0.1671, 0.1692, 0.1711, 0.1731, 0.1752, 0.1771, 0.1792, 0.1812,
                    0.1831, 0.1852, 0.1871, 0.1891, 0.1912, 0.1931, 0.1952, 0.1971, 0.1991, 0.2012,
                    0.2031, 0.2051, 0.2072, 0.2092
                ]
            ]
        ]
    );
    Ok(())
}

#[test]
fn flash_attn_varlen() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let q = Tensor::arange(0u32, 3 * 2 * 64, &device)?
        .to_dtype(DType::F16)?
        .reshape((3, 2, 64))?;
    let k = (&q / 400.)?;
    let v = (&q / 500.)?;
    let q = (&q / 300.)?;

    let seqlens_q = Tensor::new(&[0u32, 2u32], &device)?;
    // let seqlens_k: Tensor = Tensor::new(&[0u32, 3u32], &device)?;

    let ys = {
        let q = q.transpose(0, 1)?;
        let k = k.transpose(0, 1)?;
        let v = v.transpose(0, 1)?;
        flue_flash_attn_v3::flash_attn_varlen(
            &q, &k, &v, &seqlens_q, &seqlens_q, 2, 2, 0.5, false, false,
        )?
        .transpose(0, 1)?
    };
    let ys = ys.to_dtype(DType::F32)?;

    assert_eq!(ys.dims(), &[3, 2, 64]);
    assert_eq!(
        to_vec3_round(ys, 4)?,
        &[
            [
                [
                    0.0808, 0.0828, 0.0848, 0.0869, 0.0889, 0.0908, 0.0928, 0.0948, 0.0969, 0.0989,
                    0.1008, 0.1028, 0.1049, 0.1069, 0.1088, 0.1108, 0.1129, 0.1149, 0.1168, 0.1188,
                    0.1208, 0.1229, 0.1249, 0.1268, 0.1288, 0.1309, 0.1328, 0.1349, 0.1368, 0.1388,
                    0.1409, 0.1428, 0.1449, 0.1469, 0.1488, 0.1509, 0.1528, 0.1548, 0.1569, 0.1588,
                    0.1609, 0.1628, 0.1648, 0.1669, 0.1688, 0.1709, 0.1729, 0.1748, 0.1769, 0.1788,
                    0.1809, 0.1829, 0.1848, 0.1869, 0.1888, 0.1908, 0.1929, 0.1948, 0.1969, 0.1989,
                    0.2008, 0.2029, 0.205, 0.2069
                ],
                [
                    0.1071, 0.1091, 0.1111, 0.113, 0.1151, 0.1171, 0.1191, 0.1211, 0.123, 0.1251,
                    0.1271, 0.129, 0.1311, 0.1331, 0.135, 0.1371, 0.139, 0.1411, 0.1431, 0.145,
                    0.1471, 0.149, 0.1511, 0.1531, 0.155, 0.1571, 0.1591, 0.1611, 0.1631, 0.165,
                    0.1671, 0.1691, 0.1711, 0.1731, 0.175, 0.1771, 0.1791, 0.181, 0.1831, 0.1851,
                    0.1871, 0.1891, 0.191, 0.1931, 0.1951, 0.1971, 0.1991, 0.201, 0.2031, 0.2051,
                    0.2072, 0.2091, 0.2111, 0.2131, 0.2151, 0.217, 0.2191, 0.2211, 0.2231, 0.2251,
                    0.2271, 0.229, 0.2312, 0.2332
                ]
            ],
            [
                [
                    0.3765, 0.3784, 0.3804, 0.3823, 0.3843, 0.3862, 0.3884, 0.3904, 0.3923, 0.3943,
                    0.3962, 0.3984, 0.4004, 0.4023, 0.4043, 0.4063, 0.4084, 0.4104, 0.4124, 0.4143,
                    0.4163, 0.4185, 0.4204, 0.4224, 0.4243, 0.4263, 0.4285, 0.4304, 0.4324, 0.4343,
                    0.4363, 0.4385, 0.4404, 0.4424, 0.4443, 0.4463, 0.4485, 0.4504, 0.4524, 0.4543,
                    0.4563, 0.4585, 0.4604, 0.4624, 0.4644, 0.4663, 0.4683, 0.4705, 0.4724, 0.4744,
                    0.4763, 0.4783, 0.4805, 0.4824, 0.4844, 0.4863, 0.4883, 0.4905, 0.4922, 0.4946,
                    0.4966, 0.4985, 0.5005, 0.5024
                ],
                [
                    0.3816, 0.3835, 0.3855, 0.3875, 0.3894, 0.3914, 0.3936, 0.3955, 0.3975, 0.3994,
                    0.4014, 0.4036, 0.4055, 0.4075, 0.4094, 0.4114, 0.4136, 0.4155, 0.4175, 0.4194,
                    0.4214, 0.4236, 0.4255, 0.4275, 0.4294, 0.4314, 0.4336, 0.4355, 0.4375, 0.4395,
                    0.4414, 0.4436, 0.4456, 0.4475, 0.4495, 0.4514, 0.4536, 0.4556, 0.4575, 0.4595,
                    0.4614, 0.4636, 0.4656, 0.4675, 0.4695, 0.4714, 0.4734, 0.4756, 0.4775, 0.4795,
                    0.4814, 0.4834, 0.4856, 0.4875, 0.4895, 0.4915, 0.4934, 0.4956, 0.4973, 0.4998,
                    0.5015, 0.5034, 0.5054, 0.5073
                ]
            ],
            [
                [
                    0.6392, 0.6411, 0.6431, 0.6455, 0.6475, 0.6494, 0.6514, 0.6533, 0.6553, 0.6572,
                    0.6592, 0.6611, 0.6631, 0.6655, 0.6675, 0.6694, 0.6714, 0.6733, 0.6753, 0.6772,
                    0.6792, 0.6812, 0.6831, 0.6851, 0.6875, 0.6895, 0.6914, 0.6934, 0.6953, 0.6973,
                    0.6992, 0.7012, 0.7031, 0.7051, 0.7075, 0.7095, 0.7114, 0.7134, 0.7153, 0.7173,
                    0.7192, 0.7212, 0.7231, 0.7251, 0.7275, 0.7295, 0.7314, 0.7334, 0.7354, 0.7373,
                    0.7393, 0.7412, 0.7432, 0.7451, 0.7476, 0.7495, 0.7515, 0.7534, 0.7554, 0.7573,
                    0.7593, 0.7612, 0.7632, 0.7651
                ],
                [
                    0.6396, 0.6416, 0.6436, 0.646, 0.6479, 0.6499, 0.6519, 0.6538, 0.6558, 0.6577,
                    0.6597, 0.6616, 0.6636, 0.666, 0.668, 0.6699, 0.6719, 0.6738, 0.6758, 0.6777,
                    0.6797, 0.6816, 0.6836, 0.6855, 0.688, 0.6899, 0.6919, 0.6938, 0.6958, 0.6978,
                    0.6997, 0.7017, 0.7036, 0.7056, 0.708, 0.71, 0.7119, 0.7139, 0.7158, 0.7178,
                    0.7197, 0.7217, 0.7236, 0.7256, 0.728, 0.73, 0.7319, 0.7339, 0.7358, 0.7378,
                    0.7397, 0.7417, 0.7437, 0.7456, 0.748, 0.75, 0.752, 0.7539, 0.7559, 0.7578,
                    0.7598, 0.7617, 0.7637, 0.7656
                ]
            ]
        ]
    );
    Ok(())
}

#[rstest(
    head_dim => [64, 128, 256],
    seq_len => [2, 4, 9],
    use_gqa_packing => [false], // true does not make sense, as its reset to falser in the function
)]
fn flash_attn_varlen_param(head_dim: usize, seq_len: usize, use_gqa_packing: bool) -> Result<()> {
    let device = Device::new_cuda(0)?;

    // Adjust the shape so it reflects seq_len.
    // Here, we make q of shape (3, seq_len, head_dim).
    let q = Tensor::arange(0u32, (3 * seq_len * head_dim) as u32, &device)?
        .to_dtype(DType::F16)?
        .reshape((3, seq_len, head_dim))?;
    // divide by max value to have expected magnitude of error.
    let k = (&q / ((head_dim * seq_len) as f64 * 4.))?;
    let v = (&q / ((head_dim * seq_len) as f64 * 2.))?;
    let q = (&q / ((head_dim * seq_len) as f64 * 3.))?;

    // For varlen, we need start/end offsets for each “batch element.”
    // In this test, we have only 1 “batch element,” so let's do `[0, seq_len]`.
    let seqlens_q = Tensor::new(&[0u32, seq_len as u32], &device)?;
    let seqlens_k = Tensor::new(&[0u32, seq_len as u32], &device)?;

    let ys = {
        let q = q.transpose(0, 1)?;
        let k = k.transpose(0, 1)?;
        let v = v.transpose(0, 1)?;
        flue_flash_attn_v3::flash_attn_varlen(
            &q,
            &k,
            &v,
            &seqlens_q,
            &seqlens_k,
            seq_len,         // max_seqlen_q
            seq_len,         // max_seqlen_k
            0.5,             // softmax scale
            false,           // causal
            use_gqa_packing, // use_gqa_packing
        )?
        .transpose(0, 1)? // bring it back to (3, seq_len, head_dim)
    };
    let ys = ys.to_dtype(DType::F32)?;

    assert_eq!(ys.dims(), &[3, seq_len, head_dim]);
    let ys2 = {
        // reference implementation
        let q = q.unsqueeze(0)?;
        let k = k.unsqueeze(0)?;
        let v = v.unsqueeze(0)?;
        let y = fa_acausal(&q, &k, &v, 0.5)?;
        y.i(0)?.to_dtype(DType::F32)?
    };

    let diff = ys.sub(&ys2)?.abs()?.flatten_all()?.max(0)?;
    assert!(diff.to_vec0::<f32>()?.abs() < 5e-3);
    Ok(())
}
