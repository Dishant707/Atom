use std::io::{self, Write, Read};
use glam::Vec3;

// --- WRITER ---
pub struct DCDWriter<W: Write> {
    writer: W,
    n_atoms: u32,
    n_frames: u32,
    start_step: u32,
    interval: u32,
}

impl<W: Write> DCDWriter<W> {
    pub fn new(mut writer: W, n_atoms: u32) -> io::Result<Self> {
        // Validation
        if n_atoms == 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "Number of atoms must be > 0"));
        }

        // 1. Header Block
        let signature = b"CORD";
        write_fortran_block(&mut writer, |w| {
            w.write_all(signature)?; // 4 bytes
             
            // Control arrays (ICNTRL) - 20 integers
            let mut icntrl = vec![0i32; 20];
            icntrl[0] = 0; // NSET (Number of frames)
            icntrl[1] = 1; // ISTART 
            icntrl[2] = 1; // NSAVC 
            icntrl[19] = 24; // Charmm version
            
            for &val in &icntrl {
                w.write_all(&val.to_le_bytes())?;
            }
            Ok(())
        })?;

        // 2. Title Block
        write_fortran_block(&mut writer, |w| {
             let n_titles = 1i32;
             w.write_all(&n_titles.to_le_bytes())?;
             // Pad or ensure 80 chars
             let mut final_title = Vec::new();
             final_title.extend_from_slice(b"Created by Atom Engine 2025");
             while final_title.len() < 80 { final_title.push(b' '); }
             w.write_all(&final_title)
        })?;

        // 3. Atom Count Block
        write_fortran_block(&mut writer, |w| {
            w.write_all(&n_atoms.to_le_bytes())
        })?;

        Ok(Self {
            writer,
            n_atoms,
            n_frames: 0,
            start_step: 0,
            interval: 1,
        })
    }

    pub fn write_frame(&mut self, positions: &[Vec3]) -> io::Result<()> {
        if positions.len() as u32 != self.n_atoms {
             return Err(io::Error::new(io::ErrorKind::InvalidInput, "Frame atom count mismatch"));
        }

        // Unit Cell Block (Optional but often expected)
        write_fortran_block(&mut self.writer, |w| {
            let box_size = [100.0f64, 90.0, 100.0, 90.0, 90.0, 100.0];
            for &val in &box_size {
                w.write_all(&val.to_le_bytes())?;
            }
            Ok(())
        })?;

        // Coordinates
        write_fortran_block(&mut self.writer, |w| {
            for pos in positions { w.write_all(&pos.x.to_le_bytes())?; }
            Ok(())
        })?;
        write_fortran_block(&mut self.writer, |w| {
            for pos in positions { w.write_all(&pos.y.to_le_bytes())?; }
            Ok(())
        })?;
        write_fortran_block(&mut self.writer, |w| {
            for pos in positions { w.write_all(&pos.z.to_le_bytes())?; }
            Ok(())
        })?;

        self.n_frames += 1;
        self.writer.flush()?;
        Ok(())
    }
}

// --- READER ---
pub struct DCDHeader {
    pub n_frames: u32,
    pub start_step: u32,
    pub interval: u32,
    pub n_atoms: u32,
}

pub fn read_dcd<R: Read>(mut reader: R) -> io::Result<(DCDHeader, Vec<Vec<Vec3>>)> {
    // 1. Header Block
    let block = read_fortran_block_read(&mut reader)?;
    if block.len() < 84 { 
         return Err(io::Error::new(io::ErrorKind::InvalidData, "Header too short"));
    }
    
    let signature = &block[0..4];
    if signature != b"CORD" {
         return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid DCD signature"));
    }
    
    let n_frames = i32::from_le_bytes(block[4..8].try_into().unwrap()) as u32;
    let start_step = i32::from_le_bytes(block[8..12].try_into().unwrap()) as u32;
    let interval = i32::from_le_bytes(block[12..16].try_into().unwrap()) as u32;
    // 3. Atom Count (skip title which is usually block 2)
    let _title = read_fortran_block_read(&mut reader)?;
    let atom_block = read_fortran_block_read(&mut reader)?;
    let n_atoms = i32::from_le_bytes(atom_block[0..4].try_into().unwrap()) as u32;
    
    let header = DCDHeader { n_frames, start_step, interval, n_atoms };
    let mut frames = Vec::new();

    loop {
        // Peek X or UnitCell
        let res = read_fortran_block_read(&mut reader);
        if res.is_err() { break; } 
        let mut x_block = res.unwrap();
        
        // Skip UnitCell (48 bytes)
        if x_block.len() == 48 && n_atoms * 4 != 48 {
             if let Ok(next) = read_fortran_block_read(&mut reader) {
                 x_block = next;
             } else { break; }
        }
        
        let y_block = read_fortran_block_read(&mut reader)?;
        let z_block = read_fortran_block_read(&mut reader)?;
        
        // Parse
        let mut frame = Vec::with_capacity(n_atoms as usize);
        for i in 0..n_atoms as usize {
            let x = f32::from_le_bytes(x_block[i*4..(i+1)*4].try_into().unwrap());
            let y = f32::from_le_bytes(y_block[i*4..(i+1)*4].try_into().unwrap());
            let z = f32::from_le_bytes(z_block[i*4..(i+1)*4].try_into().unwrap());
            frame.push(Vec3::new(x, y, z));
        }
        frames.push(frame);
    }
    Ok((header, frames))
}


// --- HELPERS ---
fn write_fortran_block<W: Write, F>(writer: &mut W, content_fn: F) -> io::Result<()> 
where F: FnOnce(&mut dyn Write) -> io::Result<()>
{
    let mut buffer = Vec::new();
    content_fn(&mut buffer)?;
    let len = buffer.len() as i32;
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(&buffer)?;
    writer.write_all(&len.to_le_bytes())?;
    Ok(())
}

fn read_fortran_block_read<R: Read>(reader: &mut R) -> io::Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    reader.read_exact(&mut len_buf)?;
    let len = i32::from_le_bytes(len_buf) as usize;
    let mut buffer = vec![0u8; len];
    reader.read_exact(&mut buffer)?;
    reader.read_exact(&mut len_buf)?; // skip trailing
    Ok(buffer)
}
