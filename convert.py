def e2e():
    from src.training.dcc_tf import Net
    import onnx

    opset=11
    n_mics=1
    n_spk=1
    label_len=41
    chunk_size=13*10
    lookahead=0
    model = Net(n_mics,n_spk,label_len, L=32,
                enc_dim=256, num_enc_layers=10,
                dec_dim=128, dec_buf_len=13, num_dec_layers=1,
                dec_chunk_size=chunk_size, out_buf_len=4, lookahead=True)
    model.eval()
    model.exporting = True

    eg_mixed = torch.randn(1, n_mics,
                          chunk_size + lookahead)  
    eg_label = torch.randn(1, 41)

    traced_model = torch.jit.trace(model, (eg_mixed, eg_label))

    # Export the model
    output_path = os.path.join('us_'+str(opset)+'.onnx')
    torch.onnx.export(model,
                    (eg_mixed, eg_label),
                      output_path,
                      export_params=True,
                      input_names = ['x',
                                      'label',
                                      ],
                      output_names = [
                                      'filtered'
                                      ],
                      opset_version=opset,
    )
    print ('done1')
    ######################################################################3

    onnx_file='us_'+str(opset)+'.onnx'
    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)
    ort_sess = ort.InferenceSession(onnx_file)

    mixed = eg_mixed.numpy()
    label = eg_label.numpy()
    output = ort_sess.run(None, {'x': mixed,
                                 'label': label,
                                 })
    output = torch.from_numpy(output[0])

    print ('done2')

e2e()
